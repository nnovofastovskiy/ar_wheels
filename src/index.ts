import * as ort from "onnxruntime-web";

interface Detection {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    confidence: number;
    classId: number;
}

async function imageToTensor(img: HTMLImageElement, size = 640): Promise<ort.Tensor> {
    // 1️⃣ Создаём canvas для масштабирования
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d")!;
    canvas.width = size;
    canvas.height = size;
    ctx.drawImage(img, 0, 0, size, size);

    // 2️⃣ Получаем пиксели
    const { data } = ctx.getImageData(0, 0, size, size);

    // 3️⃣ Преобразуем в Float32Array [1,3,H,W]
    const floatData = new Float32Array(1 * 3 * size * size);
    for (let i = 0; i < size * size; i++) {
        floatData[i] = data[i * 4] / 255;             // R
        floatData[i + size * size] = data[i * 4 + 1] / 255; // G
        floatData[i + 2 * size * size] = data[i * 4 + 2] / 255; // B
    }

    // 4️⃣ Визуализируем, что реально идёт в модель
    // Создаём новый canvas
    const preview = document.createElement("canvas");
    preview.width = size;
    preview.height = size;
    const pctx = preview.getContext("2d")!;
    const imageData = pctx.createImageData(size, size);

    // восстанавливаем RGB из floatData
    for (let i = 0; i < size * size; i++) {
        const r = floatData[i] * 255;
        const g = floatData[i + size * size] * 255;
        const b = floatData[i + 2 * size * size] * 255;
        imageData.data[i * 4] = r;
        imageData.data[i * 4 + 1] = g;
        imageData.data[i * 4 + 2] = b;
        imageData.data[i * 4 + 3] = 255;
    }

    pctx.putImageData(imageData, 0, 0);

    // 5️⃣ Добавляем подписи и вставляем на страницу
    const caption = document.createElement("div");
    caption.textContent = `🧩 Tensor Preview (${size}×${size}) — нормализованный вход [0–1]`;
    caption.style.fontFamily = "sans-serif";
    caption.style.fontSize = "14px";
    caption.style.margin = "8px 0 4px 0";

    document.body.appendChild(caption);
    // preview.style.border = "1px solid #888";
    preview.style.width = `${Math.min(size, 640)}px`; // ограничим ширину
    preview.style.height = `${Math.min(size, 640)}px`;
    preview.style.objectFit = "contain";
    document.body.appendChild(preview);

    // 6️⃣ Вернём тензор для инференса
    return new ort.Tensor("float32", floatData, [1, 3, size, size]);
}



async function runYOLOSeg(imageUrl: string) {
    ort.env.wasm.wasmPaths = "./wasm/";
    const modelPath = "./models/best.onnx";

    const size = 640;

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = imageUrl;
    await new Promise((res) => (img.onload = res));

    // Вычисляем пропорции для letterbox (сохранение aspect ratio)
    const scale = Math.min(size / img.width, size / img.height);
    const scaledW = Math.round(img.width * scale);
    const scaledH = Math.round(img.height * scale);
    const offsetX = Math.floor((size - scaledW) / 2);
    const offsetY = Math.floor((size - scaledH) / 2);

    const session = await ort.InferenceSession.create(modelPath, {
        executionProviders: ["wasm"],
    });

    const inputTensor = await imageToTensor(img);

    const feeds: Record<string, ort.Tensor> = {};
    feeds[session.inputNames[0]] = inputTensor;

    // console.log(feeds);


    const results = await session.run(feeds);
    console.log("🧠 Выход модели:", results);

    // Обычно у YOLOSeg:
    //   outputs[0] — детекции [batch, 116, 8400]
    //   outputs[1] — прототипы масок [batch, 32, 160, 160]
    const outputNames = session.outputNames;
    const detections = results[outputNames[0]];
    const proto = results[outputNames[1]].data as Float32Array;

    console.log(results[outputNames[0]]);


    // const boxes = postprocessYOLOSeg(detections, 640, 0.25, 0.45);
    const boxes = parseDetections(
        detections,
        0.25,
        img.width,
        img.height,
        scale,
        offsetX,
        offsetY
    );
    console.log("✅ Детекции после постобработки:", boxes);
    renderBoxes(img, boxes);
}

function parseDetections(
    detections: ort.Tensor,
    threshold: number,
    originalWidth: number,
    originalHeight: number,
    scale: number,
    offsetX: number,
    offsetY: number
) {

    console.log(originalWidth, 'x', originalHeight);
    console.log(scale);
    console.log(offsetX, '-', offsetY);


    const [batch, numDets, features] = detections.dims;
    const data = detections.data;
    const boxes = [];

    for (let i = 0; i < numDets; i++) {
        const offset = i * features;
        const cx = Number(data[offset]);
        const cy = Number(data[offset + 1]);
        const w = Number(data[offset + 2]);
        const h = Number(data[offset + 3]);
        const objectness = Number(data[offset + 4]);

        // Фильтруем по objectness
        if (objectness < threshold) continue;

        // Берем максимальный класс из оставшихся вероятностей
        let maxClassScore = 0;
        let maxClassIdx = 0;

        for (let c = 5; c < features; c++) {
            const score = Number(data[offset + c]);
            if (score > maxClassScore) {
                maxClassScore = score;
                maxClassIdx = c - 5;
            }
        }

        const confidence = objectness * maxClassScore;
        if (confidence < threshold) continue;

        // Преобразуем обратно в координаты оригинального видео
        const x1 = ((cx - w / 2) - offsetX) / scale;
        const y1 = ((cy - h / 2) - offsetY) / scale;
        const x2 = ((cx + w / 2) - offsetX) / scale;
        const y2 = ((cy + h / 2) - offsetY) / scale;

        boxes.push({
            x1: Math.max(0, x1),
            y1: Math.max(0, y1),
            x2: Math.min(originalWidth, x2),
            y2: Math.min(originalHeight, y2),
            confidence,
            classId: maxClassIdx
        });
    }

    return boxes;
}

// /** Постобработка YOLO-выхода + NMS */
// function postprocessYOLOSeg(
//     data: Float32Array,
//     size: number,
//     confThres = 0.25,
//     iouThres = 0.45
// ): Detection[] {
//     const numDet = data.length / 116; // YOLOv8-Seg output [116,8400]
//     const boxes: Detection[] = [];

//     for (let i = 0; i < numDet; i++) {
//         const x1 = data[i * 116 + 0];
//         const y1 = data[i * 116 + 1];
//         const w = data[i * 116 + 2];
//         const x2 = x1 + w;
//         const h = data[i * 116 + 3];
//         const y2 = y1 + h;
//         const conf = data[i * 116 + 4];
//         if (conf < confThres) continue;

//         // ищем максимальный класс
//         let classId = 0;
//         let maxProb = 0;
//         for (let j = 5; j < 85; j++) {
//             const p = data[i * 116 + j];
//             if (p > maxProb) {
//                 maxProb = p;
//                 classId = j - 5;
//             }
//         }

//         const score = conf * maxProb;
//         if (score < confThres) continue;

//         boxes.push({ x1, y1, x2, y2, confidence, classId });
//     }

//     // применяем NMS
//     return boxes;
// }

function renderBoxes(img: HTMLImageElement, boxes: Detection[]) {
    const canvas = document.createElement("canvas");
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(img, 0, 0);
    ctx.strokeStyle = "lime";
    ctx.lineWidth = 2;
    ctx.font = "16px sans-serif";
    ctx.fillStyle = "lime";

    const classIdMap = new Map<number, number>();


    for (const box of boxes) {
        classIdMap.set(box.classId, (classIdMap.get(box.classId) || 0) + 1);
        // if (box.score < 0.95) continue;  // порог для отображения
        const x = (box.x1 + box.x2) / 2;
        const y = (box.y1 + box.y2) / 2;
        const w = Math.abs(box.x1 - box.x2);
        const h = Math.abs(box.y1 - box.y2);
        // const [x, y, w, h] = [box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1];
        ctx.strokeRect(x, y, w, h);
        ctx.fillText(`id:${box.classId} ${box.confidence.toFixed(2)}`, x, y - 4);
    }
    console.log(classIdMap);

    document.body.appendChild(canvas);
    const link = document.createElement("a");
    link.textContent = "💾 Сохранить результат";
    link.href = canvas.toDataURL("image/png");
    link.download = "yolo_output.png";
    document.body.appendChild(link);
}

// ▶️ Запуск
runYOLOSeg("./images/inputs/car.png");
