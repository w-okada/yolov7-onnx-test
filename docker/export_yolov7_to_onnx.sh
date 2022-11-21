#!/bin/bash

img_sizes=(320 640 1280 1920)
for size in ${img_sizes[@]}; do
    echo $size
    python3 export.py --weights ./yolov7-tiny.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size $size $size --max-wh $size && mv ./yolov7-tiny.onnx /models/yolov7-tiny-${size}.onnx
done

ls -lha /models

