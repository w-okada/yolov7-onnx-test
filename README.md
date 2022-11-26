YOLOv7-ONNX-TEST
----
This repository is intended only to evaluate the performance of the yolov7 onnx model. 
Model type is only Tiny.

1. python onnx (cpu)

# Prerequisite
This repository use docker.

# Experiment

## python onnx (cpu)
This code is used.
```py

start_time = time.time()
for i in range(10):
    outputs = session.run(outname, inp)[0]
elapsed_time = time.time() - start_time
print(img_size, 'fin. avr time:', (elapsed_time / 10) * 1000, "msec")

```
## Result
Result on  "Intel(R) Core(TM) i9-9900KF CPU @ 3.60GHz"

| Size      | from       | python onnx (cpu) |
| --------- | ---------- | ----------------- |
| 320x320   | Official   | 9.98msec          |
| 640x640   | Official   | 39.14msec         |
| 1280x1280 | Official   | 183.11msec        |
| 1920x1920 | Official   | 411.53msec        |
| 256x320   | PINTO (*1) | (*2)              |
| 256x480   | PINTO      | 11.26msec         |
| 256x640   | PINTO      | 14.43msec         |
| 348x640   | PINTO      | 23.73msec         |
| 480x640   | PINTO      | 28.63msec         |
| 640x640   | PINTO      | 44.12msec         |
| 736x1280  | PINTO      | 100.65msec        |

(*1) PINTO model includes postprocess. [link](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/307_YOLOv7)
(*2) seems not to work correctly.



# Operations
You can reproduce the experiment according to the following way.

## build docker
```
$ npm run build:docker
```

## export onnx model
Only `yolox_nano.pth` is supported.
```
$ npm run export:yolov7
```

## run python onnx
```
$ npm run run:yolov7_onnx
$ npm run run:yolov7_onnx_pinto
```
Wait for a while. Then you can see the output on the termianl.

## Misc
![image](https://user-images.githubusercontent.com/48346627/204105532-0df4ba51-54da-4bb3-9a5a-ff4c5f66181f.png)


[web demo](https://w-okada.github.io/yolov7-onnx-test/) only for onnx.

for tfjs is under construction. (stacked. technical problem occured.)

https://user-images.githubusercontent.com/48346627/204106193-757ebeac-5eb1-4f85-b0aa-dc0a57454ead.mp4



# Reference
1. https://github.com/WongKinYiu/yolov7
1. https://github.com/PINTO0309/PINTO_model_zoo/tree/main/307_YOLOv7

