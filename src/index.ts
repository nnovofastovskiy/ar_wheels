import * as ort from 'onnxruntime-web';
const timerEl = document.getElementById('timer') as HTMLSpanElement;
const startTime = new Date();
timerEl.innerText = `Start in ${startTime.getHours()}:${startTime.getMinutes()}:${startTime.getSeconds()}:${startTime.getMilliseconds()}`;
// === Загрузить метки ===
const labels: string[] = await fetch('./models/labels.json').then(r => r.json());

// === Константы модели ===
const modelPath = './models/best_seg_nms.onnx';
const inputSize = 640;

// === Сессия ONNX ===
const session = await ort.InferenceSession.create(modelPath, {
    executionProviders: ['wasm'],
});
const canvas = document.getElementById('outCanvas') as HTMLCanvasElement;
const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;

// === Генерация случайных цветов для классов ===
function classColor(cls: number) {
    const hash = Array.from(labels[cls]).reduce((h, c) => h + c.charCodeAt(0), 0);
    const r = (hash * 37) % 255;
    const g = (hash * 67) % 255;
    const b = (hash * 97) % 255;
    return `rgb(${r},${g},${b})`;
}

// === Загрузка изображения ===
// document.getElementById('image').addEventListener('change', async (e) => {
const img = document.getElementById('image') as HTMLImageElement;
// img.src = URL.createObjectURL(e.target.files[0]);
// await img.decode();

canvas.width = img.width;
canvas.height = img.height;

ctx.drawImage(img, 0, 0, img.width, img.height);

const inputTensor = await preprocess(img);
const output = await session.run({ images: inputTensor });

const output0 = output[session.outputNames[0]].data as Float32Array; // [1,300,38]
const output1 = output[session.outputNames[1]].data as Float32Array; // [1,32,160,160]

drawResults(output0, output1, ctx, img.width, img.height);
// });

// === Препроцессинг ===
async function preprocess(img: CanvasImageSource) {
    const tmp = document.createElement('canvas');
    const tctx = tmp.getContext('2d') as CanvasRenderingContext2D;
    tmp.width = inputSize;
    tmp.height = inputSize;
    tctx.drawImage(img, 0, 0, inputSize, inputSize);
    const imgData = tctx.getImageData(0, 0, inputSize, inputSize);
    const data = new Float32Array(inputSize * inputSize * 3);
    for (let i = 0; i < inputSize * inputSize; i++) {
        data[i] = imgData.data[i * 4] / 255.0;
        data[i + inputSize * inputSize] = imgData.data[i * 4 + 1] / 255.0;
        data[i + 2 * inputSize * inputSize] = imgData.data[i * 4 + 2] / 255.0;
    }
    return new ort.Tensor('float32', data, [1, 3, inputSize, inputSize]);
}

// === Рисуем результаты ===
function drawResults(
    output0: Float32Array,
    output1: Float32Array,
    ctx: CanvasRenderingContext2D,
    imgW: number,
    imgH: number
) {
    const detections = output0.length / 38;
    const maskProtos = new Float32Array(output1);
    const maskW = 160, maskH = 160;

    const scaleX = imgW / 640;
    const scaleY = imgH / 640;


    for (let i = 0; i < detections; i++) {
        const o = i * 38;
        const x1 = output0[o + 0] * scaleX;
        const y1 = output0[o + 1] * scaleY;
        const x2 = output0[o + 2] * scaleX;
        const y2 = output0[o + 3] * scaleY;
        const conf = output0[o + 4];
        const cls = Math.floor(output0[o + 5]);
        if (conf < 0.25) continue;

        const color = classColor(cls);
        const colorMatch = color.match(/\d+/g);
        if (!colorMatch) continue;
        const maskCoeffs = output0.slice(o + 6, o + 38);

        // === Генерация маски ===
        const mask = new Float32Array(maskW * maskH).fill(0);
        for (let m = 0; m < 32; m++) {
            for (let p = 0; p < maskW * maskH; p++) {
                mask[p] += maskProtos[m * maskW * maskH + p] * maskCoeffs[m];
            }
        }

        // === Сигмоида ===
        for (let p = 0; p < mask.length; p++) mask[p] = 1 / (1 + Math.exp(-mask[p]));

        // === Маска в canvas ===
        const maskCanvas = document.createElement('canvas');
        maskCanvas.width = maskW;
        maskCanvas.height = maskH;
        const mctx = maskCanvas.getContext('2d') as CanvasRenderingContext2D;
        const maskImg = mctx.createImageData(maskW, maskH);

        for (let p = 0; p < mask.length; p++) {
            const val = mask[p];
            const alpha = Math.min(255, val * 255 * 0.6); // прозрачность

            maskImg.data[p * 4 + 0] = parseInt(colorMatch[0]);
            maskImg.data[p * 4 + 1] = parseInt(colorMatch[1]);
            maskImg.data[p * 4 + 2] = parseInt(colorMatch[2]);
            maskImg.data[p * 4 + 3] = alpha;
        }
        mctx.putImageData(maskImg, 0, 0);

        // Масштабирование и отображение маски
        ctx.globalAlpha = 0.7;
        ctx.drawImage(maskCanvas, 0, 0, imgW, imgH);
        ctx.globalAlpha = 1.0;

        // === Bounding box ===
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // === Подпись ===
        ctx.fillStyle = color;
        ctx.font = '16px sans-serif';
        const text = `${labels[cls]} ${(conf * 100).toFixed(1)}%`;
        const textW = ctx.measureText(text).width;
        ctx.fillRect(x1, y1 - 18, textW + 6, 18);
        ctx.fillStyle = '#fff';
        ctx.fillText(text, x1 + 3, y1 - 5);
    }
}


const endTime = new Date();
timerEl.innerText += `\nEnds in ${endTime.getHours()}:${endTime.getMinutes()}:${endTime.getSeconds()}:${endTime.getMilliseconds()}\nTotal time: ${endTime.getTime() - startTime.getTime()} ms   `;
