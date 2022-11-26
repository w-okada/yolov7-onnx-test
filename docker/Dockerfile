FROM debian:bullseye-slim as yolox

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    python3-pip \
    git \
    curl \
    wget \
    emacs

RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

RUN apt-get install -y libgl1-mesa-dev

RUN pip install onnx
RUN pip install onnx-simplifier
RUN pip install onnxruntime
RUN pip install pandas
RUN pip install opencv-python
RUN pip install tqdm
RUN pip install pyyaml
RUN pip install matplotlib
RUN pip install seaborn
RUN pip install scipy
RUN pip install onnx2tf
RUN pip install tensorflow
RUN pip install tensorflowjs
RUN pip install nvidia-pyindex
RUN pip install onnx-graphsurgeon
RUN pip install sng4onnx


RUN git clone --depth 1 https://github.com/WongKinYiu/yolov7 

WORKDIR /yolov7
RUN wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt

COPY /export_yolov7_to_onnx.sh /yolov7
COPY /run_yolov7_onnx.py /yolov7
COPY /run_yolov7_onnx_pinto.py /yolov7

#ENTRYPOINT ["/bin/bash", "export_onnx.sh"]
