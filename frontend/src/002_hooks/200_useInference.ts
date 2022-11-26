import { useRef, useState } from "react"
import { PERFORMANCE_ALL_SPAN, PERFORMANCE_INFER_SPAN, TARGET_CANVAS_ID, TARGET_IMAGE_ID, TARGET_VIDEO_ID, TEMPORARY_CANVAS_ID } from "../const";
import { MediaType } from "./100_useFrontendManager";
import { InferenceSession, Tensor } from 'onnxruntime-web';

import * as tf from '@tensorflow/tfjs';

export const EngineType = {
    onnx: "onnx",
    // tfjs: "tfjs"
} as const
export type EngineType = typeof EngineType[keyof typeof EngineType]

export const InputShape = {
    "256x320": "256x320",
    "256x480": "256x480",
    "256x640": "256x6400",
    "320x320": "320x320",
    "384x640": "384x640",
    "416x416": "416x416",
    "480x640": "480x640",
    "640x640": "640x640",
    "736x1280": "736x1280",
    "1088x1920": "1088x1920",
    "1280x1280": "1280x1280",
    "1920x1920": "1920x1920",
} as const
export type InputShape = typeof InputShape[keyof typeof InputShape]

const ScoreThreshold = 0.3
// const NMSDropOverlapThreshold = 0.55
const PerformancCounter_num = 10

type BoundingBox = {
    startX: number,
    startY: number,
    endX: number,
    endY: number,
    classIdx: number,
    score: number,
}


const names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']

export type InferenceState = {
    processId: number
    engineType: EngineType
    inputShape: InputShape
}

export type InferenceStateAndMethod = InferenceState & {
    startProcess: (type: MediaType, inputResolution: [number, number], processId: number) => Promise<void>
    stopProcess: () => Promise<void>
    setEnginType: (val: EngineType) => void
    setInputShape: (val: InputShape) => void
}


export const useInference = (): InferenceStateAndMethod => {

    const processIdRef = useRef<number>(0)
    const [processId, setProcessId] = useState<number>(processIdRef.current)
    const [engineType, setEnginType] = useState<EngineType>("onnx")
    const [inputShape, setInputShape] = useState<InputShape>("256x320")

    const perfCounterInferenceRef = useRef<number[]>([])
    const perfCounterAllRef = useRef<number[]>([])

    const _updatePerfCounterInference = (counter: number) => {
        perfCounterInferenceRef.current.push(counter)
        while (perfCounterInferenceRef.current.length > PerformancCounter_num) {
            perfCounterInferenceRef.current.shift()
        }
        const perfCounterInferenceAvr = perfCounterInferenceRef.current.reduce((prev, cur) => {
            return prev + cur
        }) / perfCounterInferenceRef.current.length;
        (document.getElementById(PERFORMANCE_INFER_SPAN) as HTMLSpanElement).innerText = `${perfCounterInferenceAvr.toFixed(2)}`
    }
    const _updatePerfCounterAll = (counter: number) => {
        perfCounterAllRef.current.push(counter)
        while (perfCounterAllRef.current.length > PerformancCounter_num) {
            perfCounterAllRef.current.shift()
        }
        const perfCounterAllAvr = perfCounterAllRef.current.reduce((prev, cur) => {
            return prev + cur
        }) / perfCounterAllRef.current.length;
        (document.getElementById(PERFORMANCE_ALL_SPAN) as HTMLSpanElement).innerText = `${perfCounterAllAvr.toFixed(2)}`
    }

    const _inferWithONNX = async (session: InferenceSession, canvas: HTMLCanvasElement, inputShapeArray: number[], ratio: number) => {
        const validBox: BoundingBox[] = []
        // generate input tensor
        //// get source image data (input tensor size)
        const ctx = canvas.getContext("2d")!
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
        const [redArray, greenArray, blueArray] = new Array(new Array<number>(), new Array<number>(), new Array<number>());
        //// get RGB Data
        for (let i = 0; i < imageData.data.length; i += 4) {
            redArray.push(imageData.data[i]);
            greenArray.push(imageData.data[i + 1]);
            blueArray.push(imageData.data[i + 2]);
            // skip data[i + 3] to filter out the alpha channel
        }
        //// Transpose
        const transposedData = blueArray.concat(greenArray).concat(redArray);
        const float32Data = new Float32Array(3 * inputShapeArray[0] * inputShapeArray[1]);
        for (let i = 0; i < transposedData.length; i++) {
            float32Data[i] = transposedData[i] / 255.0
        }
        //// Generate
        const inputTensor = new Tensor("float32", float32Data, [1, 3, inputShapeArray[0], inputShapeArray[1]]);

        // Process YOLOX
        const inname = session.inputNames[0]
        const feeds = {};
        // @ts-ignore
        feeds[inname] = inputTensor
        const perfCounterInference_start = performance.now()
        const results = await session.run(feeds)
        const perfCounterInference_end = performance.now()
        const perfCounterInference = perfCounterInference_end - perfCounterInference_start
        _updatePerfCounterInference(perfCounterInference)

        const output = results.output
        // console.log(output, output.dims)

        for (let i = 0; i < output.data.length; i += output.dims[1]) {
            const startX = (output.data[i + 1] as number) / ratio
            const startY = (output.data[i + 2] as number) / ratio
            const endX = (output.data[i + 3] as number) / ratio
            const endY = (output.data[i + 4] as number) / ratio
            const classIdx = output.data[i + 5] as number
            const score = output.data[i + 6] as number
            validBox.push({
                startX, startY, endX, endY, classIdx, score
            })
        }
        // console.log(validBox)
        return validBox
    }


    const _inferWithTFJS = async (session: tf.GraphModel, canvas: HTMLCanvasElement, _inputShapeArray: number[], _ratio: number) => {
        //!*****************************************************************!/
        //!** I don't know why but I couldn't run YOLOv7 official model.  **!/
        //!** (output shape is '(2) [100, 63883]'. What is this??)        **!/
        //!*****************************************************************!/

        console.log("SESSIOM:1::", session.inputs)
        console.log("SESSIOM:2::", session.inputNodes)
        console.log("SESSIOM:3::", session.outputs)
        console.log("SESSIOM:4::", session.outputNodes)


        const validBox: BoundingBox[] = []

        const t = tf.browser.fromPixels(canvas).expandDims(0).cast("float32");
        // Process YOLOX
        const perfCounterInference_start = performance.now()
        // @ts-ignore
        // let prediction = (await session.executeAsync(t)) as tf.Tensor  // for official
        let prediction = (await session.executeAsync(t)) as [tf.Tensor, tf.Tensor] // for pinto
        const perfCounterInference_end = performance.now()
        const perfCounterInference = perfCounterInference_end - perfCounterInference_start
        _updatePerfCounterInference(perfCounterInference)

        // console.log("PREDICTION:1::", prediction)
        // const data1 = prediction[0].arraySync()
        // const data2 = prediction[1].arraySync()
        // prediction[0].dispose()
        // prediction[1].dispose()
        // console.log("PREDICTION:3::", data1)
        // console.log("PREDICTION:3::", data2)

        return validBox
    }
    const stopProcess = async () => {
        processIdRef.current = 0
        setProcessId(processIdRef.current)
    }

    const startProcess = async (type: MediaType, inputResolution: [number, number], processId: number) => {
        if (inputResolution[0] == 0) {
            console.log("input resolution is 0")
            return
        }
        // Params for GUI
        //// ProcessId
        processIdRef.current = processId
        setProcessId(processIdRef.current)
        const thisProcessId = processId

        //// inputShape
        const inputShapeArray = inputShape.split("x").map(x => { return Number(x) })

        //// Model File
        const modelFilePath = engineType == "onnx" ?
            `./models/yolov7_tiny_${inputShape}.onnx` : `./models/tfjs_yolov7_tiny_${inputShape}/model.json`
        // const modelFilePath = engineType == "onnx" ?
        //     `./models/yolov7_tiny_${inputShape}.onnx` : `./models/yolov7-pinto-tiny_post_480x640/model.json`

        // Target Video/Image and Target Canvas
        const targetId = type === "image" ? TARGET_IMAGE_ID : TARGET_VIDEO_ID
        const target = document.getElementById(targetId) as HTMLImageElement | HTMLCanvasElement
        const targetCanvas = document.getElementById(TARGET_CANVAS_ID) as HTMLCanvasElement
        targetCanvas.width = inputResolution[0]
        targetCanvas.height = inputResolution[1]

        // Tmp Canvas (setup width/height to input tensor shape)
        const tmpCanvas = document.getElementById(TEMPORARY_CANVAS_ID) as HTMLCanvasElement
        tmpCanvas.height = inputShapeArray[0]
        tmpCanvas.width = inputShapeArray[1]

        // Calc image size on Tmp Canvas
        const ratio = Math.min(inputShapeArray[1] / inputResolution[0], inputShapeArray[0] / inputResolution[1])
        const width = inputResolution[0] * ratio
        const height = inputResolution[1] * ratio

        console.log("Process Configuration")
        console.log(`
            Original_Image_Size: (w:${inputResolution[0]}, h:${inputResolution[1]}), 
            Tensor_Shape: (h:${inputShapeArray[0]}, w:${inputShapeArray[1]}),
            Image Ratio and Size on Tensor: (ratio:${ratio}, w:${width}, h:${height})
            `)

        // Create Session
        const session = engineType == "onnx" ?
            await InferenceSession.create(modelFilePath)
            :
            await tf.loadGraphModel(modelFilePath);


        // Main Process Function
        const process = async () => {
            const perfCounterAll_start = performance.now()
            // Copy snapshot of target video/image to target canvas (same size)
            const targetCtx = targetCanvas.getContext("2d")!
            targetCtx.drawImage(target, 0, 0, targetCanvas.width, targetCanvas.height)

            // Clear tmp canvas with grey (input tensor shape size)
            const ctx = tmpCanvas.getContext("2d")!
            ctx.fillStyle = "rgb(114, 114, 114)";
            ctx.fillRect(0, 0, tmpCanvas.width, tmpCanvas.height)

            // draw snapshot image to tmp canvas (image size resized with ratio)
            ctx.drawImage(target, 0, 0, width, height)


            // Inference
            let validBox: BoundingBox[] = []
            if (engineType == "onnx") {
                validBox = await _inferWithONNX(session as InferenceSession, tmpCanvas, inputShapeArray, ratio)
            } else {
                validBox = await _inferWithTFJS(session as tf.GraphModel, tmpCanvas, inputShapeArray, ratio)
            }

            // Draw BoundingBox
            validBox.forEach((x) => {
                if (x.score < ScoreThreshold) {
                    return
                }
                targetCtx.beginPath();
                targetCtx.strokeStyle = 'white';
                targetCtx.lineWidth = 3;
                targetCtx.rect(x.startX, x.startY, x.endX - x.startX, x.endY - x.startY);
                targetCtx.stroke();
                targetCtx.fillStyle = "red";
                targetCtx.font = "bold 14px 'Segoe Print', san-serif";
                targetCtx.fillText(`${names[x.classIdx]}, ${(x.score * 100).toFixed(1)}%`, x.startX, x.startY)

            })
            const perfCounterAll_end = performance.now()
            const perfCounterAll = perfCounterAll_end - perfCounterAll_start
            _updatePerfCounterAll(perfCounterAll)

            if (thisProcessId === processIdRef.current) {
                // console.log(`next process loop (this:${thisProcessId}, current:${processIdRef.current})`)
                requestAnimationFrame(process)
            } else {
                // console.log(`stop process loop (this:${thisProcessId}, current:${processIdRef.current})`)
            }
        }
        requestAnimationFrame(process)
    }

    const returnValue = {
        processId,
        engineType,
        inputShape,
        startProcess,
        stopProcess,
        setEnginType,
        setInputShape,
    };
    return returnValue;
};


