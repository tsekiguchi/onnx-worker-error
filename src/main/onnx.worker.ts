import { parentPort, workerData } from 'worker_threads';
import * as ort from 'onnxruntime-node';
import { BatchInferenceResponse } from './index.js';
import { SimpleTokenizer } from './SimpleTokenizer.js';

// const VISION_MODEL_Q8_PATH = workerData.VISION_MODEL_Q8_PATH;
const TEXT_MODEL_Q8_PATH = workerData.TEXT_MODEL_Q8_PATH;

// let visionModel: ort.InferenceSession | null = null;run d
let textModel: ort.InferenceSession | null = null;

if (parentPort) {
	parentPort.on('message', async (req) => {
		if (!req) parentPort?.postMessage('NO MESSAGE RECEIVED');

		if (req.action === 'generateTextEmbeddings') {
			const embeddings = await generateTextEmbeddings(req.data);
			parentPort?.postMessage(embeddings);
		}
	});
}

// async function loadVisionModel(): Promise<ort.InferenceSession> {
// 	if (visionModel) return visionModel;

// 	visionModel = await ort.InferenceSession.create(VISION_MODEL_Q8_PATH, {
// 		executionProviders: [{ name: 'cpu' }],
// 		graphOptimizationLevel: 'all'
// 	});

// 	return visionModel;
// }

async function loadTextModel(): Promise<ort.InferenceSession> {
	if (textModel) return textModel;

	textModel = await ort.InferenceSession.create(TEXT_MODEL_Q8_PATH, {
		executionProviders: [{ name: 'cpu' }],
		graphOptimizationLevel: 'all'
	});

	return textModel;
}

// async function generateImageEmbeddings(
// 	arrayBuffers: ArrayBuffer[]
// ): Promise<BatchInferenceResponse> {
// 	// if (!visionModel) visionModel = await loadVisionModel();
// 	const visionModel = await loadVisionModel();

// 	const float32Arrays = arrayBuffers.map((arrayBuffer) => new Float32Array(arrayBuffer));
// 	const totalLength = float32Arrays.reduce((acc, arr) => acc + arr.length, 0);

// 	// Create a new Float32Array of the required length.
// 	const result = new Float32Array(totalLength);

// 	// Copy each input array into the result using the .set method.
// 	let offset = 0;
// 	for (const arr of float32Arrays) {
// 		result.set(arr, offset);
// 		offset += arr.length;
// 	}

// 	// Create an ONNX Runtime Tensor using the array
// 	const float32Tensor = new ort.Tensor('float32', result, [float32Arrays.length, 3, 224, 224]);

// 	// Magic
// 	const { image_embeddings } = await visionModel.run({ image: float32Tensor });

// 	// Extract the data to send it across the IPC
// 	const f32Data = (await image_embeddings.getData()) as Float32Array;

// 	return {
// 		arrayBuffer: f32Data.buffer as ArrayBuffer,
// 		batchSize: image_embeddings.dims[0]
// 	};
// }

async function generateTextEmbeddings(inputs: string | string[]): Promise<BatchInferenceResponse> {
	const textModel = await loadTextModel();

  const tokenizer = new SimpleTokenizer();
	const text_inputs = tokenizer.tokenize(inputs);

	const { text_embeddings } = await textModel.run({ text: text_inputs });
	if (!text_embeddings) throw Error('No embedding returned from text model');

	const embeddings = (await text_embeddings.getData()) as Float32Array;

	return {
		arrayBuffer: embeddings.buffer as ArrayBuffer,
		batchSize: text_embeddings.dims[0]
	};
}
