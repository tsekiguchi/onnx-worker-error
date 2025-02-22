import { app, shell, BrowserWindow } from 'electron';
import path, { join } from 'path';
import { electronApp, optimizer, is } from '@electron-toolkit/utils';
import icon from '../../resources/icon.png?asset';
import serve from 'electron-serve';
import windowStateManager from 'electron-window-state';
import FS from 'fs';
import { Worker } from 'worker_threads';
//@ts-ignore "?modulePath" wildcard is recognized by electron-vite bundler
import transformersWorkerPath from './onnx.worker?modulePath';
import { SimpleTokenizer } from './SimpleTokenizer.js';
import { downloadFile } from './downloadHandler.js';
import * as ort from 'onnxruntime-node';
import { ipcMain } from 'electron/main';

const serveURL = serve({ directory: join(__dirname, '..', 'renderer') });

export interface BatchInferenceResponse {
	arrayBuffer: ArrayBuffer;
	batchSize: number;
}

export let mainWindow: BrowserWindow;
function createWindow(): void {
	// window state is useful when application is closed & opened again
	// especially useful in dev, when app is reloaded on main/preload file changes
	const windowState = windowStateManager({
		defaultWidth: 800,
		defaultHeight: 600
	});

	// Create the browser window.
	mainWindow = new BrowserWindow({
		width: windowState.width,
		height: windowState.height,
		x: windowState.x,
		y: windowState.y,
		show: false,
		autoHideMenuBar: true,
		...(process.platform === 'linux' ? { icon } : {}),
		webPreferences: {
			preload: join(__dirname, '../preload/index.mjs'),
			sandbox: false
		}
	});

	// initialize window state
	windowState.manage(mainWindow);
	// and save state when exiting
	mainWindow.on('close', () => {
		windowState.saveState(mainWindow);
	});

	mainWindow.on('ready-to-show', () => {
		mainWindow.show();
	});

	mainWindow.webContents.setWindowOpenHandler((details) => {
		shell.openExternal(details.url);
		return { action: 'deny' };
	});
	// HMR for renderer base on electron-vite cli.
	// Load the remote URL for development or the local html file for production.
	if (is.dev) {
		loadVite();
		mainWindow.webContents.openDevTools();
		// mainWindow.loadURL(import.meta.env.MAIN_VITE_ELECTRON_RENDERER_URL)
	} else {
		// mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
		serveURL(mainWindow);
	}
}

// this is needed to prevent blank screen when dev electron loads
function loadVite(): void {
	mainWindow.loadURL(import.meta.env.MAIN_VITE_ELECTRON_RENDERER_URL).catch((e) => {
		console.log('Error loading URL, retrying', e);
		setTimeout(() => {
			loadVite();
		}, 200);
	});
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(async () => {
	// Set app user model id for windows
	electronApp.setAppUserModelId('com.electron');

	// Default open or close DevTools by F12 in development
	// and ignore CommandOrControl + R in production.
	// see https://github.com/alex8088/electron-toolkit/tree/master/packages/utils
	app.on('browser-window-created', (_, window) => {
		optimizer.watchWindowShortcuts(window);
	});

	createWindow();

	app.on('activate', function () {
		// On macOS it's common to re-create a window in the app when the
		// dock icon is clicked and there are no other windows open.
		if (BrowserWindow.getAllWindows().length === 0) createWindow();
	});
});

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', () => {
	if (process.platform !== 'darwin') {
		app.quit();
	}
});

// In this file you can include the rest of your app"s specific main process
// code. You can also put them in separate files and require them here.

ipcMain.handle('onnx-worker-test', async () => {
	try {
		return await testOnnxRuntimeWorker();
	} catch (error) {
		return { error: (error as Error).message };
	}
});

ipcMain.handle('onnx-node-test', async () => {
	return await testOnnxRuntimeNode();
});

const VISION_MODEL_Q8_PATH = path.join(app.getPath('userData'), 'vision_model_q8_batch.onnx');
const TEXT_MODEL_Q8_PATH = path.join(app.getPath('userData'), 'text_model_q8_batch.onnx');

const TEXT_MODEL_Q8_URL =
	'https://huggingface.co/recallapp/MobileCLIP-B-LT-OpenCLIP/resolve/main/onnx/text_model_q8_batch.onnx?download=true';

const VISION_MODEL_Q8_URL =
	'https://huggingface.co/recallapp/MobileCLIP-B-LT-OpenCLIP/resolve/main/onnx/vision_model_q8_batch.onnx?download=true';

async function downloadModels(): Promise<void> {
	if (!FS.existsSync(VISION_MODEL_Q8_PATH))
		await downloadFile(VISION_MODEL_Q8_URL, VISION_MODEL_Q8_PATH, 'Vision Model');
	if (!FS.existsSync(TEXT_MODEL_Q8_PATH))
		await downloadFile(TEXT_MODEL_Q8_URL, TEXT_MODEL_Q8_PATH, 'Text Model');
}

async function testOnnxRuntimeWorker(): Promise<ArrayBuffer> {
	await downloadModels();
	const worker = new Worker(transformersWorkerPath, {
		workerData: {
			VISION_MODEL_Q8_PATH,
			TEXT_MODEL_Q8_PATH
		}
	});

	const tokenizer = new SimpleTokenizer();

	const input = tokenizer.tokenize('A man and his dog at the beach');

	const text_embeddings = await new Promise<BatchInferenceResponse>((resolve) => {
		worker.postMessage({ action: 'generateTextEmbeddings', data: input });

		worker.on('message', (response) => {
			console.log('worker response', response);
			resolve(response);
		});
	});

	const data = text_embeddings.arrayBuffer;

	return data;
}

async function testOnnxRuntimeNode(): Promise<ArrayBuffer> {
	await downloadModels();

	const textModel = await ort.InferenceSession.create(TEXT_MODEL_Q8_PATH, {
		executionProviders: [{ name: 'cpu' }],
		graphOptimizationLevel: 'all'
	});

	const tokenizer = new SimpleTokenizer();

	const input = tokenizer.tokenize('A man and his dog at the beach');

	const { text_embeddings } = await textModel.run({ text: input });
	if (!text_embeddings) throw Error('No embedding returned from text model');

	console.log('Text Embeddings', text_embeddings);

	const data = (await text_embeddings.getData()) as Float32Array;

	return data.buffer;
}
