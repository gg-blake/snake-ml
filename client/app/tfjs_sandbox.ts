import * as tf from '@tensorflow/tfjs';
const m = 5
const tensorBuf = tf.eye(m).expandDims(0).tile([3, 1, 1]).bufferSync();
tensorBuf.set(4, ...[0, 0, 1])
console.log(tensorBuf.toTensor());

for (let i = 0; i <= 4; i++) {
    const firstBit = (i >> 1) & 1;  // Get the second-to-last bit
    const secondBit = i & 1;        // Get the last bit

    console.log(`i = ${i}, firstBit = ${firstBit}, secondBit = ${secondBit}`);
}
