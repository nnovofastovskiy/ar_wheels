// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { InferenceSession } from "onnxruntime-web";

// see also advanced usage of importing ONNX Runtime Web:
// https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js/importing_onnxruntime-web
const ort = require('onnxruntime-web');

const loadModelStatus = document.getElementById('model-load-state');

// use an async context to call onnxruntime functions.
async function main() {
    try {
        // create a new session and load the specific model.
        //
        // the model in this example contains a single MatMul node
        // it has 2 inputs: 'a'(float32, 3x4) and 'b'(float32, 4x3)
        // it has 1 output: 'c'(float32, 3x3)
        ort.env.wasm.wasmPaths = "./wasm/";
        const session: InferenceSession = await ort.InferenceSession.create('./models/best_seg_nms.onnx');

        const image = document.getElementById('image');
        if (!image || !(image instanceof HTMLImageElement)) {
            throw new Error('Could not find image element');
        }
        setStatus('image preprocessed into tensor.');
        const tensor = await preprocessImage(image);
        // console.log(tensor.dims); // [1, 3, 640, 640]
        // console.log(tensor.type); // 'float32'
        const { output0, output1 } = await session.run({ images: tensor });

        // YOLO обычно возвращает один выход с детекциями
        // Формат: [batch, num_detections, 5 + num_classes]
        // где 5 = [cx, cy, w, h, objectness]
        console.log("output0.dims = ", output0.dims);
        console.log("output1.dims = ", output1.dims);

        setStatus(`data of result tensor`);
        // Допустим, у тебя уже есть результаты модели:
        const proto = output1.data as Float32Array; // Float32Array(1*32*160*160)
        const det = output0.data as Float32Array;  // Float32Array(1*59*8400)

        console.log("output0 = ", det);
        console.log("output1 = ", proto);


        // Возьмём первую детекцию с уверенностью > 0.25
        const numPreds = 8400;
        const numAttrs = 59;

        const [batch, numDets, features] = output0.dims;
        const data = output0.data;
        const boxes: {
            x: number,
            y: number,
            w: number,
            h: number,
            confidence: number,
            classId: number
        }[] = [];

        const threshold = 0.15;

        const objectnessEs: number[] = [];
        const classIds: number[] = [];

        for (let i = 0; i < numDets; i++) {
            const offset = i * features;
            const x = data[offset] as number;
            const y = data[offset + 1] as number;
            const w = data[offset + 2] as number;
            const h = data[offset + 3] as number;
            const objectness = data[offset + 4] as number;
            const classId = data[offset + 5] as number;

            objectnessEs.push(objectness);
            classIds.push(classId);

            // Фильтруем по objectness
            if (objectness < threshold) continue;

            // Берем максимальный класс из оставшихся вероятностей
            let maxClassScore = 0;
            let maxClassIdx = 0;

            for (let c = 6; c < features; c++) {
                const score = data[offset + c] as number;
                if (score > maxClassScore) {
                    maxClassScore = score;
                    maxClassIdx = c - 6;
                }
            }

            const confidence = objectness * maxClassScore;
            if (confidence < threshold) continue;

            boxes.push({
                x,
                y,
                w,
                h,
                confidence,
                classId: maxClassIdx
            });
        }

        console.log(boxes);
        console.log(objectnessEs);
        console.log(objectnessEs);


        // for (let i = 0; i < 2; i = i + numAttrs) {
        //     const offset = i;
        //     const conf = det[offset + 4];
        //     console.log(conf, i);

        //     if (conf < 0.85) continue;

        //     const x = det[offset];
        //     const y = det[offset + 1];
        //     const w = det[offset + 2];
        //     const h = det[offset + 3];
        //     const coeff = det.slice(offset + 5 + 22, offset + 59); // если 22 класса

        //     const box = [x - w / 2, y - h / 2, x + w / 2, y + h / 2];

        //     const maskImg = reconstructMask(proto, coeff, box, 640);

        //     // Нарисуем маску поверх изображения:
        //     const canvas = document.getElementById('outCanvas');
        //     if (!canvas || !(canvas instanceof HTMLCanvasElement)) {
        //         throw new Error('Could not find outCanvas element');
        //     }
        //     const ctx = canvas.getContext('2d');
        //     if (!ctx) {
        //         throw new Error('Could not get canvas context');
        //     }
        //     ctx.putImageData(maskImg, 0, 0);
        //     // break;
        // }

    } catch (e) {
        setError(`failed to inference ONNX model: ${e}.`);
    }
}

main();

async function preprocessImage(image: HTMLImageElement) {
    // 1. Нарисуем в canvas и изменим размер до 640x640

    const size = 640;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
        throw new Error('Could not get canvas context');
    }
    ctx.drawImage(image, 0, 0, size, size);

    // 2. Получим данные пикселей
    const imageData = ctx.getImageData(0, 0, size, size);
    const { data } = imageData; // Uint8ClampedArray [r,g,b,a,r,g,b,a,...]

    // 3. Преобразуем в Float32Array и нормализуем [0,1]
    const float32Data = new Float32Array(3 * size * size);
    for (let i = 0; i < size * size; i++) {
        float32Data[i] = data[i * 4] / 255.0;        // R
        float32Data[i + size * size] = data[i * 4 + 1] / 255.0;  // G
        float32Data[i + 2 * size * size] = data[i * 4 + 2] / 255.0;  // B
    }

    // 4. Создаем тензор с нужной формой [1,3,640,640]
    const tensor = new ort.Tensor('float32', float32Data, [1, 3, size, size]);
    return tensor;
}

function setStatus(message: string) {
    const loadModelStatus = document.getElementById('state');
    if (loadModelStatus) {
        loadModelStatus.innerText = message;
    }
}

function setError(message: string) {
    const errorStatus = document.getElementById('error');
    if (errorStatus) {
        errorStatus.innerText = message;
    }
}

function reconstructMask(protoData: Float32Array, coeff: Float32Array, box: number[], inputSize = 640) {
    const [maskC, maskH, maskW] = [32, 160, 160]
    const mask = new Float32Array(maskH * maskW)

    // === 1️⃣ Суммируем 32 прототипа с весами ===
    for (let i = 0; i < maskH * maskW; i++) {
        let v = 0
        for (let j = 0; j < maskC; j++) {
            v += coeff[j] * protoData[j * maskH * maskW + i]
        }
        mask[i] = 1 / (1 + Math.exp(-v)) // sigmoid
    }

    // === 2️⃣ Апсемплим с 160→640 (простое билинейное приближение) ===
    const upsampled = new Float32Array(inputSize * inputSize)
    for (let y = 0; y < inputSize; y++) {
        for (let x = 0; x < inputSize; x++) {
            const srcX = x * (maskW / inputSize)
            const srcY = y * (maskH / inputSize)
            const x0 = Math.floor(srcX)
            const x1 = Math.min(x0 + 1, maskW - 1)
            const y0 = Math.floor(srcY)
            const y1 = Math.min(y0 + 1, maskH - 1)
            const wx = srcX - x0
            const wy = srcY - y0
            const top = mask[y0 * maskW + x0] * (1 - wx) + mask[y0 * maskW + x1] * wx
            const bottom = mask[y1 * maskW + x0] * (1 - wx) + mask[y1 * maskW + x1] * wx
            upsampled[y * inputSize + x] = top * (1 - wy) + bottom * wy
        }
    }

    // === 3️⃣ Вырезаем по bbox ===
    const [x1, y1, x2, y2] = box.map(v => Math.max(0, Math.min(inputSize, v)))
    const maskRGBA = new Uint8ClampedArray(inputSize * inputSize * 4)
    for (let y = 0; y < inputSize; y++) {
        for (let x = 0; x < inputSize; x++) {
            const val = upsampled[y * inputSize + x]
            const inside = (x >= x1 && x <= x2 && y >= y1 && y <= y2)
            const alpha = inside && val > 0.5 ? 120 : 0
            const i = (y * inputSize + x) * 4
            maskRGBA[i + 0] = 0
            maskRGBA[i + 1] = 255
            maskRGBA[i + 2] = 0
            maskRGBA[i + 3] = alpha
        }
    }

    return new ImageData(maskRGBA, inputSize, inputSize)
}
