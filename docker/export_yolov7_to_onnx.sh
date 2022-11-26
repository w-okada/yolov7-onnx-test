#!/bin/bash

img_sizes=(416x416 320x320 640x640 1280x1280 1920x1920 256x320 256x480 256x640 384x640 480x640 736x1280 1088x1920)

#img_sizes=(416x416)

for size in ${img_sizes[@]}; do
    echo $size
    hw_array=() 
    hw=$(echo $size|tr "x" "\n")
    for element in $hw; do
        hw_array=(${hw_array[@]} $element)
    done
    python3 export.py --weights ./yolov7-tiny.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size ${hw_array[0]} ${hw_array[1]} --max-wh ${hw_array[1]} && mv ./yolov7-tiny.onnx /models/yolov7_tiny_${size}.onnx

    onnx2tf -i /models/yolov7_tiny_${size}.onnx -o /models/tflite_yolov7_tiny_${size} --output_signaturedefs

    tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default  --saved_model_tags=serve /models/tflite_yolov7_tiny_${size} /models/tfjs/tfjs_yolov7_tiny_${size}

    # tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default  --saved_model_tags=serve --output_node_names "PartitionedCall/model/tf.concat_18/concat" /models/tflite_yolov7_tiny_${size} /models/tfjs/tfjs_yolov7_tiny_${size}



    # Note:
    # There is no information about what is max-wh of yolov7. 
    # But regarding yolov5, there is. 
    #   https://github.com/ultralytics/yolov5/issues/1875
    # Max-wh may be for nms to decide box size? 
    # I cannot understand the effect.
    # https://github.com/WongKinYiu/yolov7/blob/main/models/experimental.py
    #   dis = category_id.float() * self.max_wh
    #   nmsbox = boxes + dis
    # nmsbox is boxes broadcasted with dis(= some value * max_wh)???
    # .... For now, set them to the maximum width and height.
done


onnx2tf -i /models/pinto/yolov7-tiny_post_480x640.onnx -o /models/tflite_pinto_yolov7_tiny_post_480x640 --output_signaturedefs

tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default  --saved_model_tags=serve /models/tflite_pinto_yolov7_tiny_post_480x640 /models/tfjs/pinto_yolov7_tiny_post_480x640