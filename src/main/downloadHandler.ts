import { net } from 'electron/main';
import fs from 'fs';

export async function downloadFile(url: string, dest: string, fileName: string): Promise<void> {
	return new Promise<void>((resolve, reject) => {
		if (!net.online) reject('Application is offline');

		const request = net.request(url);

		request.on('response', (response) => {
			// Get the total size from the 'content-length' header (if available)
			const totalBytes = parseInt((response.headers['content-length'] as string) || '0', 10);
			// const parsedUrl = new URL(url);
			// Use the pathname and take its basename
			// const fileName = PATH.basename(parsedUrl.pathname);

			let currentBytes = 0;

			const tmpName = dest + '.tmp';

			const fileStream = fs.createWriteStream(tmpName);

			response.on('data', (chunk) => {
				fileStream.write(chunk);
				currentBytes += chunk.length;
				if (totalBytes) {
					const progress = (currentBytes / totalBytes) * 100;

					console.log(`Downloading model: ${fileName}...`, progress);
				} else {
					console.log(`Downloaded ${currentBytes} bytes`);
				}
			});

			response.on('end', () => {
				fileStream.end();
				fs.renameSync(tmpName, dest);
				resolve();
			});

			fileStream.on('error', (err) => {
				fs.unlink(dest, () => reject(err));
			});
		});

		request.on('error', reject);
		request.end();
	});
}
