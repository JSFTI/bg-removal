import {
    env,
    AutoModel,
    AutoProcessor,
    RawImage,
    PreTrainedModel,
    Processor
} from "@huggingface/transformers";
import { createCanvas } from '@napi-rs/canvas';
import express from 'express';
import type { Request, Response } from 'express';
import multer from 'multer';
import { readdirSync, writeFileSync } from 'fs'

const app = express();
const port = process.env.PORT || 3001;

const storage = multer.memoryStorage();
const upload = multer({
    storage,
    limits: {
        fileSize: 10 * 1024 * 1024
    },
    fileFilter: (req, file, cb) => {
        // Accept images only
        if (file.mimetype.startsWith('image/')) {
            cb(null, true);
        } else {
            cb(new Error('Only image files are allowed'));
        }
    },
});

let modelState: {
    model: PreTrainedModel | null;
    processor: Processor | null;
    initialized: boolean;
} = {
    model: null,
    processor: null,
    initialized: false,
};

const MODEL_ID = 'Xenova/modnet';

async function ensureModelLoaded() {
    if (modelState.initialized) return;

    env.allowLocalModels = false;

    modelState.model = await AutoModel.from_pretrained(MODEL_ID, {
        dtype: "fp32",
    });

    modelState.processor = await AutoProcessor.from_pretrained(MODEL_ID, {
        config: {
            do_normalize: true,
            do_pad: false,
            do_rescale: true,
            do_resize: true,
            image_mean: [0.5, 0.5, 0.5],
            feature_extractor_type: "ImageFeatureExtractor",
            image_std: [1, 1, 1],
            resample: 2,
            rescale_factor: 0.00392156862745098,
            size: { width: 1024, height: 1024 },
        }
    });

    modelState.initialized = true;
}

app.post('/bg-removal', upload.single('file'), async (req: Request, res: Response) => {
    if (req.header('Authorization') !== `Bearer ${process.env.API_KEY}`) {
        res.status(401).send('Unauthorized');
        return;
    }

    const file = req.file;

    await ensureModelLoaded();

    // @ts-expect-error
    const img = await RawImage.fromBlob(new Blob([file!.buffer]));
    try {
        const startTime = new Date().getTime();
        const { pixel_values } = await modelState.processor!(img);

        // Predict alpha matte
        const { output } = await modelState.model!({ input: pixel_values, });

        // Resize mask back to original size
        const maskData = (
            await RawImage.fromTensor(output[0].mul(255).to("uint8")).resize(
                img.width,
                img.height,
            )
        ).data;

        const canvas = createCanvas(img.width, img.height);
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext("2d");
        if (!ctx) throw new Error("Could not get 2d context");

        const imageData = ctx.createImageData(img.width, img.height);

        const channels = img.channels;

        for (let i = 0; i < img.width * img.height; i++) {
            if (channels === 3) {
                // RGB input
                imageData.data[i * 4] = img.data[i * 3];         // R
                imageData.data[i * 4 + 1] = img.data[i * 3 + 1]; // G
                imageData.data[i * 4 + 2] = img.data[i * 3 + 2]; // B
                imageData.data[i * 4 + 3] = maskData[i];         // A (from mask)
            } else if (channels === 4) {
                // RGBA input
                imageData.data[i * 4] = img.data[i * 4];         // R
                imageData.data[i * 4 + 1] = img.data[i * 4 + 1]; // G
                imageData.data[i * 4 + 2] = img.data[i * 4 + 2]; // B
                imageData.data[i * 4 + 3] = maskData[i];         // A (from mask)
            }
        }

        ctx.putImageData(imageData, 0, 0);
        const pngData = await canvas.encode("png");

        const endTime = new Date().getTime();

        writeFileSync(`./files/${endTime - startTime}.png`, Buffer.from(pngData));

        res.send(pngData);
    } catch (error) {
        console.error("Error processing image:", error);
        throw new Error("Failed to process image");
    }
});

app.get('/', (req, res) => {
    const files = readdirSync("./files");

    res.send(files.map((file) => parseInt(file) / 1000));
});

app.listen(port);
