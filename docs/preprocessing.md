#### COCO Format
docker run --rm -u $(id -u):$(id -g) \
    -v `pwd`:/workspace \
    -it waikatodatamining/image-dataset-converter:latest\
  idc-convert -l INFO \
    from-adams-od \
      -l INFO \
      -i "adams_whole/*.report" \
    discard-negative \
    sub-images \
      --row_height 720 \
      --col_width 720 \
      --overlap_right 80 \
      --overlap_bottom 80 \
      --pad_width 800 \
      --pad_height 800 \
      -p \
      -e \
    to-coco-od \
      -o coco_sub_split/ \
      --split_name train val test\
      --split_ratios 70 15 15

#### YOLO Format
docker run --rm -u $(id -u):$(id -g) \
    -v `pwd`:/workspace \
    -it waikatodatamining/image-dataset-converter:latest\
  idc-convert -l INFO \
    from-adams-od \
      -l INFO \
      -i "adams_whole/*.report" \
    discard-negative \
    sub-images \
      --row_height 576 \
    --col_width 576 \
    --overlap_right 64 \
    --overlap_bottom 64 \
    --pad_width 640 \
    --pad_height 640 \
      -p \
      -e \
    to-yolo-od \
    -o yolo_sub_split/ \
    --labels labels.txt\
    --labels_csv labels.csv \
    --split_name train val test\
    --split_ratios 70 15 15