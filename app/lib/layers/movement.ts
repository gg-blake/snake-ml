import * as tf from '@tensorflow/tfjs';
import { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';
import '@tensorflow/tfjs-backend-webgl';
import GameLayer, { GameLayerConfig, IntermediateLayer } from './gamelayer';
import { generatePlaneIndices } from '../util';

const rotateBatch: IntermediateLayer<[tf.Tensor3D, tf.Tensor2D, tf.Tensor1D], tf.Tensor3D> = (B, T, C, direction, planeIndices, theta) => tf.tidy(() => {
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
    return _result
});

type MovementInputs = [
    tf.Tensor2D, // Position
    tf.Tensor3D, // Direction
    tf.Tensor2D, // Sequential Input
    tf.Tensor1D // Active
];

export default class Movement extends GameLayer {
    private planeIndices: tf.Tensor2D;

    constructor(config: LayerArgs, gameConfig: GameLayerConfig) {
        super(config, gameConfig);
        this.planeIndices = generatePlaneIndices(this.C);
    }

    call(inputs: MovementInputs): [tf.Tensor2D, tf.Tensor3D] {
        return tf.tidy(() => {
            const planeIndices = this.planeIndices.reverse(1).concat(this.planeIndices, 0);
            const indices = planeIndices.gather(inputs[2].argMax(1));
            const nextDirections = rotateBatch(this.B, this.T, this.C, inputs[1], indices, inputs[2].max(1).mul(Math.PI / 2));
            const nextVelocity: tf.Tensor2D = nextDirections.slice([0, 0], [this.B, 1]).squeeze([1]);
            const nextPositions: tf.Tensor2D = inputs[0].add(nextVelocity.mul(1).mul(inputs[3].expandDims(-1).tile([1, this.C])));
            return [ nextPositions, nextDirections ];
        })
    }

    computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
        return [[this.B, this.C], [this.B, this.C, this.C]];
    }

    static get className() {
        return 'Movement';
    }
}

tf.serialization.registerClass(Movement);