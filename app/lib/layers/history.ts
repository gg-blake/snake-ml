import * as tf from '@tensorflow/tfjs';
import { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';
import '@tensorflow/tfjs-backend-webgl';
import GameLayer, { GameLayerConfig } from './gamelayer';

type HistoryInputs = [
    tf.Tensor2D, // Position
    tf.Tensor1D, // Target Index
    tf.Tensor3D // History
];

export default class History extends GameLayer {
    constructor(config: LayerArgs, gameConfig: GameLayerConfig) {
        super(config, gameConfig);
    }

    call(inputs: HistoryInputs): tf.Tensor3D {
        const S = this.startingLength;
        return tf.tidy(() => {
            const cutoffMask = tf.range(0, this.T).expandDims().tile([this.B, 1]).less(inputs[1].expandDims(-1).tile([1, this.T]).add(S)).cast('int32'); // (B, T)
            const cutoffKeep = tf.concat([inputs[0].expandDims(1), inputs[2]], 1).slice([0, 0, 0], [this.B, this.T, this.C]); // (B, T, C)
            const nextHistory = cutoffKeep.div(cutoffMask.expandDims(1).tile([1, this.C, 1]).transpose([0, 2, 1])) as tf.Tensor3D; // (B, T, C)
            return nextHistory;
        })
    }

    computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
        return [this.B, this.T, this.C];
    }

    static get className() {
        return 'History';
    }
}

tf.serialization.registerClass(History);