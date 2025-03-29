import * as tf from '@tensorflow/tfjs';

function main() {
    const rand = tf.randomUniform([6, 6], -1, 1);
    const probs = tf.softmax(rand, 1) as tf.Tensor2D;
    probs.print()
    const samples = tf.multinomial(probs, 1);
    samples.print();
}

main();