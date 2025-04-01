import * as tf from '@tensorflow/tfjs';
import { signedPlaneAngle } from '../util';

function main() {
    const a = tf.tensor2d([[-0.926087 , -0.0010695, 0.3773086], [0.3770043 , 0.0376348 , 0.9254466]]);
    const b = tf.tensor2d([[0.3770043 , 0.0376348 , 0.9254466], [-0.926087 , -0.0010695, 0.3773086]]);
    const c = tf.tensor2d([[1, -1, 2], [1, -1, 2]]);
    const angles = signedPlaneAngle(a, b, c);
    angles.print();
}

main();