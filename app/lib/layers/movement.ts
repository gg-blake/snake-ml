import * as tf from '@tensorflow/tfjs';
import { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';
import '@tensorflow/tfjs-backend-webgl';
import GameLayer, * as GL from './gamelayer';
import { dotProduct, generatePlaneIndices } from '../util';

const rotateBatch = (
    config: GL.Config, 
    direction: GL.Direction<tf.Tensor>, 
    planeIndices: tf.Tensor2D, 
    theta: tf.Tensor1D,
    epsilon: number = 1e-8
): GL.Direction<tf.Tensor> => tf.tidy(() => {
    const [ B, T, C ] = config.batchInputShape! as number[];
    // direction: (B, C, C)
    // planeIndices: (B, 2)
    // theta: (B,)

    const _identity: tf.Tensor3D = tf.eye(C).expandDims().tile([B, 1, 1]);
    const _indicesAugmented1 = tf.range(0, B).expandDims(-1).tile([1, 2]).expandDims(-1).concat(planeIndices.expandDims(-1), -1);
    const _rotationSquareIndices = planeIndices.concat(planeIndices.reverse(1), 1).expandDims(-1);
    const _indicesAugmented2 = _indicesAugmented1.tile([1, 2, 1]).concat(_rotationSquareIndices, 2).reshape([4 * B, 3]).cast("int32");
    const _trig = tf.stack([tf.cos(theta), tf.cos(theta), tf.sin(theta).neg(), tf.sin(theta)]).transpose().reshape([4 * B]);
    const _rotation: tf.Tensor3D = tf.tensorScatterUpdate(_identity, _indicesAugmented2, _trig);
    
    const _result = tf.matMul<tf.Tensor3D>(_rotation, direction, false, true);

    // Values close to integer with difference of epsilon stick to closest integer value
    

    return _result as tf.Tensor3D
});

type MovementInputs = [
    GL.Position<tf.Tensor>, // Position
    GL.Direction<tf.Tensor>, // Direction
    tf.Tensor2D, // Controls
    GL.Active<tf.Tensor> // Active
];

export default class Movement extends GameLayer {
    private planeIndices: tf.Tensor2D;

    constructor(config: LayerArgs & GL.Config) {
        super(config, Movement.className);
        const [ B, T, C ] = this.config.batchInputShape! as number[];
        this.planeIndices = generatePlaneIndices(C);
    }

    call(inputs: MovementInputs): [tf.Tensor2D, tf.Tensor3D] {
        const [ B, T, C ] = this.config.batchInputShape! as number[];
        return tf.tidy(() => {
            const planeIndices = this.planeIndices.reverse(-1).concat(this.planeIndices, 0); // (2 * (C - 1), 2)
            const indices = planeIndices.gather(inputs[2].argMax(1)); // (B, 2)
            const nextDirections = rotateBatch(this.config, inputs[1], indices, tf.ones([B]).mul(-Math.PI / 8));
            const nextVelocity: tf.Tensor2D = nextDirections.slice([0, 0], [B, 1]).squeeze([1]);
            const nextPositions: tf.Tensor2D = inputs[0].add(nextVelocity.mul(1).mul(inputs[3].expandDims(-1).tile([1, C])));
            return [ nextPositions, nextDirections ];
        })
    }

    computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
        const [ B, T, C ] = this.config.batchInputShape! as number[];
        return [[B, C], [B, C, C]];
    }

    static get className() {
        return 'Movement';
    }
}

tf.serialization.registerClass(Movement);