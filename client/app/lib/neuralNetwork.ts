import * as ort from 'onnxruntime-web';

class NeuralNetwork {
    private inFeatures: number;
    private hiddenFeatures: number;
    private outFeatures: number;
    private weights1: ort.Tensor;
    private weights2: ort.Tensor;
    private session: ort.InferenceSession | undefined;

    constructor(inFeatures: number, hiddenFeatures: number, outFeatures: number) {
        this.inFeatures = inFeatures;
        this.hiddenFeatures = hiddenFeatures;
        this.outFeatures = outFeatures;

        // Initialize weights as tensors (assuming random initialization here for demonstration purposes)
        this.weights1 = new ort.Tensor('float32', new Float32Array(this.inFeatures * this.hiddenFeatures).map(() => Math.random()), [this.inFeatures, this.hiddenFeatures]);
        this.weights2 = new ort.Tensor('float32', new Float32Array(this.hiddenFeatures * this.outFeatures).map(() => Math.random()), [this.hiddenFeatures, this.outFeatures]);

        // Explicitly set the session to undefined to satisfy TypeScript's strict property initialization checks
        this.session = undefined;
    }

    private relu(tensor: ort.Tensor): ort.Tensor {
        const data = tensor.data as Float32Array;
        const reluData = new Float32Array(data.length);
        for (let i = 0; i < data.length; i++) {
            reluData[i] = Math.max(0, data[i]);
        }
        return new ort.Tensor('float32', reluData, tensor.dims);
    }

    public async forward(inputs: ort.Tensor): Promise<ort.Tensor> {
        // Create a session if it doesn't exist
        if (!this.session) {
            const randomBuffer = new Uint8Array(1024).map(() => Math.floor(Math.random() * 256)); // Random ONNX buffer
            this.session = await ort.InferenceSession.create(randomBuffer.buffer); // Create session from random ONNX buffer
        }

        // Input -> Hidden Layer
        const matMul1 = await this.session.run({
            input: inputs,
            weights: this.weights1
        });

        const hiddenLayer = this.relu(matMul1['output']);

        // Hidden Layer -> Output Layer
        const matMul2 = await this.session.run({
            input: hiddenLayer,
            weights: this.weights2
        });

        const output = this.relu(matMul2['output']);
        return output;
    }
}



async function main() {
    let net = new NeuralNetwork(3, 5, 2);
    const inFeatures = 3;
    const hiddenFeatures = 5;
    const outFeatures = 2;
    net = new NeuralNetwork(inFeatures, hiddenFeatures, outFeatures);

    net = new NeuralNetwork(3, 5, 2);
    const inputTensor = new ort.Tensor('float32', new Float32Array([1, 2, 3]), [3, 1]);
    try {
        const outputTensor = await net.forward(inputTensor);
        
    } catch (error) {
        console.log(error);
    }
}


(async () => {
    await main();
})(); 
