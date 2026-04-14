FROM waikatodatamining/pytorch-yolo26:8.4.16_cuda12.6

COPY --from=waikatodatamining/image-dataset-converter:latest \
    /opt/image-dataset-converter/lib/python3.12/site-packages \
    /opt/yolo26/lib/python3.12/site-packages

COPY yolo26_predict_tiled_redis /usr/local/bin/
COPY predict_tiled_redis.py /opt/yolo26-extra/

RUN chmod +x /usr/local/bin/yolo26_predict_tiled_redis