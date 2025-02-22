<script lang="ts">
	import { writable } from "svelte/store";

  const response = writable<string>('')

	const handleWorkerClick = async () => {
		const arrayBuffer = await window.electron.ipcRenderer.invoke('onnx-worker-test');
    const f32 = new Float32Array(arrayBuffer)
    response.set(f32.toString())
	};

	const handleNodeClick = async () => {
		const arrayBuffer = await window.electron.ipcRenderer.invoke('onnx-node-test');
    const f32 = new Float32Array(arrayBuffer)
    console.log('response received', f32)
    response.set(f32.toString())
	};
</script>

<h1>Welcome to SvelteKit</h1>
<p>Visit <a href="https://kit.svelte.dev">kit.svelte.dev</a> to read the documentation</p>

<button on:click={handleWorkerClick}> Click here to run Onnx in a Worker </button>

<button on:click={handleNodeClick}> Click here to run Onnx in Node </button>

<p>Embeddings</p>
<p>{$response}</p>
