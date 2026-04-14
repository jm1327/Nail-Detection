import io
import traceback
import logging
from datetime import datetime
from typing import List, Optional
from PIL import Image

import torch
from opex import ObjectPredictions, ObjectPrediction, BBox, Polygon
from rdh import Container, MessageContainer, create_parser, configure_redis, run_harness, log
from predict_common import load_model, ModelParams

from idc.api import merge_polygons as idc_merge_polygons, ObjectDetectionData
from idc.imgaug.filter._sub_images_utils import generate_regions
from wai.common.adams.imaging.locateobjects import LocatedObjects, LocatedObject
from wai.common.geometry import Point as WaiPoint, Polygon as WaiPolygon


# ---------------------------------------------------------------------------
# opex <-> idc 转换（用于调用 idc_merge_polygons）
# ---------------------------------------------------------------------------
def _opex_to_located_objects(opex_preds: List[ObjectPrediction]) -> LocatedObjects:
    lobjs = []
    for pred in opex_preds:
        lobj = LocatedObject(
            x=pred.bbox.left,
            y=pred.bbox.top,
            width=pred.bbox.right - pred.bbox.left,
            height=pred.bbox.bottom - pred.bbox.top,
        )
        lobj.metadata["score"] = pred.score
        lobj.metadata["label"] = pred.label
        pts = pred.polygon.points
        if len(pts) >= 3:
            lobj.set_polygon(WaiPolygon(*(WaiPoint(x, y) for x, y in pts)))
        lobjs.append(lobj)
    return LocatedObjects(lobjs)


def _located_objects_to_opex(lobjs: LocatedObjects) -> List[ObjectPrediction]:
    result = []
    for lobj in lobjs:
        label = lobj.metadata.get("label", "unknown")
        score = float(lobj.metadata.get("score", 1.0))
        bbox = BBox(
            left=lobj.x, top=lobj.y,
            right=lobj.x + lobj.width,
            bottom=lobj.y + lobj.height,
        )
        if lobj.has_polygon:
            pts = [[x, y] for x, y in zip(lobj.get_polygon_x(), lobj.get_polygon_y())]
        else:
            pts = [
                [bbox.left, bbox.top], [bbox.right, bbox.top],
                [bbox.right, bbox.bottom], [bbox.left, bbox.bottom],
            ]
        result.append(ObjectPrediction(score=score, label=label, bbox=bbox, polygon=Polygon(points=pts)))
    return result


# ---------------------------------------------------------------------------
# 核心推理函数
# ---------------------------------------------------------------------------
def predict_tiled_opex(model_params: ModelParams, pred_id: str, img: Image.Image,
                       confidence_threshold: float = 0.25,
                       classes: Optional[List[str]] = None,
                       augment: bool = False,
                       col_width: int = 576,
                       row_height: int = 576,
                       overlap_right: int = 0,
                       overlap_bottom: int = 0,
                       merge_adjacent_polygons: bool = True) -> ObjectPredictions:

    img_w, img_h = img.size
    logger = logging.getLogger(__name__)

    # 1. 用原版 generate_regions 生成 tile 坐标（行为与 meta-sub-images 完全一致）
    regions = generate_regions(
        width=img_w, height=img_h,
        row_height=row_height, col_width=col_width,
        overlap_right=overlap_right, overlap_bottom=overlap_bottom,
        partial=False,
        logger=logger,
    )

    # 2. 裁切所有 tile
    tiles = []
    for r in regions:
        x0, y0 = r.x, r.y
        x1 = min(x0 + r.w, img_w)
        y1 = min(y0 + r.h, img_h)
        tile = img.crop((x0, y0, x1, y1))
        tiles.append((tile, x0, y0, x1 - x0, y1 - y0))  # (图, x偏移, y偏移, 实际宽, 实际高)

    classes_set = set(classes) if classes else set(model_params.names.values())

    # 3. 批量推理（一次 GPU forward）
    tile_imgs = [t[0] for t in tiles]
    with torch.no_grad():
        all_preds = model_params.model.predict(source=tile_imgs, augment=augment, verbose=False)

    # 4. 把子图坐标映射回原图（对应 transfer_region 逻辑）
    all_opex_preds: List[ObjectPrediction] = []
    for pred, (_, x_off, y_off, tile_w, tile_h) in zip(all_preds, tiles):
        for box in pred.boxes:
            box = box.to("cpu")
            conf = float(box.conf)
            if conf < confidence_threshold:
                continue
            label = model_params.names[float(box.cls)]
            if label not in classes_set:
                continue

            xyxyn = box.xyxyn.numpy()[0]
            # tile 内像素坐标 + 偏移 = 原图坐标
            left   = int(xyxyn[0] * tile_w) + x_off
            top    = int(xyxyn[1] * tile_h) + y_off
            right  = int(xyxyn[2] * tile_w) + x_off
            bottom = int(xyxyn[3] * tile_h) + y_off
            # 裁掉超出原图边界的部分
            left   = max(0, min(left,   img_w - 1))
            top    = max(0, min(top,    img_h - 1))
            right  = max(0, min(right,  img_w))
            bottom = max(0, min(bottom, img_h))
            if right <= left or bottom <= top:
                continue

            bbox = BBox(left=left, top=top, right=right, bottom=bottom)
            poly = Polygon(points=[
                [left, top], [right, top],
                [right, bottom], [left, bottom],
            ])
            all_opex_preds.append(ObjectPrediction(
                score=conf, label=label, bbox=bbox, polygon=poly,
            ))

    # 5. 合并重叠预测（直接调用 idc merge_polygons，与 meta-sub-images 行为一致）
    dummy_img_bytes = io.BytesIO()
    img.save(dummy_img_bytes, format="JPEG")
    od_item = ObjectDetectionData(
        image_name="tmp.jpg",
        data=dummy_img_bytes.getvalue(),
        annotation=_opex_to_located_objects(all_opex_preds),
    )
    if merge_adjacent_polygons:
        od_item = idc_merge_polygons(od_item)
    merged = _located_objects_to_opex(od_item.annotation if od_item.has_annotation() else LocatedObjects())

    return ObjectPredictions(id=pred_id, timestamp=str(datetime.now()), objects=merged)


# ---------------------------------------------------------------------------
# Redis harness
# ---------------------------------------------------------------------------
def process_image(msg_cont):
    config = msg_cont.params.config
    try:
        start_time = datetime.now()
        if config.verbose:
            log("process_image - start processing image")

        img = Image.open(io.BytesIO(msg_cont.message['data']))

        preds = predict_tiled_opex(
            model_params=config.model_params,
            pred_id=str(start_time),
            img=img,
            confidence_threshold=config.confidence_threshold,
            classes=config.classes,
            augment=config.augment,
            col_width=config.col_width,
            row_height=config.row_height,
            overlap_right=config.overlap_right,
            overlap_bottom=config.overlap_bottom,
            merge_adjacent_polygons=config.merge_adjacent_polygons,
        )

        msg_cont.params.redis.publish(msg_cont.params.channel_out, preds.to_json_string())

        if config.verbose:
            log("process_image - published to: %s" % msg_cont.params.channel_out)
            ms = int((datetime.now() - start_time).total_seconds() * 1000)
            log("process_image - finished in: %d ms" % ms)

    except KeyboardInterrupt:
        msg_cont.params.stopped = True
    except:
        log("process_image - failed: %s" % traceback.format_exc())


def main(args=None):
    parser = create_parser('Yolo26 - Tiled Prediction (Redis)',
                           prog="yolo26_predict_tiled_redis", prefix="redis_")
    parser.add_argument('--model',                   metavar="FILE",   type=str,   required=True)
    parser.add_argument('--device',                  metavar="DEVICE", type=str,   default="cuda")
    parser.add_argument('--confidence_threshold',    metavar="0-1",    type=float, default=0.25)
    parser.add_argument('--classes',                 nargs='*',        type=str)
    parser.add_argument('--augment',                 action='store_true')
    parser.add_argument('--col_width',               type=int,         default=576)
    parser.add_argument('--row_height',              type=int,         default=576)
    parser.add_argument('--overlap_right',           type=int,         default=0)
    parser.add_argument('--overlap_bottom',          type=int,         default=0)
    parser.add_argument('--merge_adjacent_polygons', action='store_true',
                        help='Merge adjacent/overlapping polygons after reassembly')
    parser.add_argument('--verbose',                 action='store_true')
    parsed = parser.parse_args(args=args)

    print("Loading model (%s): %s" % (parsed.device, parsed.model))
    model_params = load_model(parsed.model, device=parsed.device)

    config = Container()
    config.model_params            = model_params
    config.confidence_threshold    = parsed.confidence_threshold
    config.classes                 = parsed.classes
    config.augment                 = parsed.augment
    config.col_width               = parsed.col_width
    config.row_height              = parsed.row_height
    config.overlap_right           = parsed.overlap_right
    config.overlap_bottom          = parsed.overlap_bottom
    config.merge_adjacent_polygons = parsed.merge_adjacent_polygons
    config.verbose                 = parsed.verbose

    params = configure_redis(parsed, config=config)
    run_harness(params, process_image)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())