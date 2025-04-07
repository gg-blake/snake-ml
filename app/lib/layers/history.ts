import * as tf from '@tensorflow/tfjs';
import { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';
import '@tensorflow/tfjs-backend-webgl';
import GameLayer, * as GL from './gamelayer';
import { ClassName } from '@tweakpane/core';

type HistoryInputs = [
    GL.Position<tf.Tensor>, // Position
    GL.TargetIndices<tf.Tensor>, // Target Index
    GL.History<tf.Tensor> // History
];

export default class History extends GameLayer {
    constructor(config: LayerArgs & GL.Config) {
        super(config, History.className);
    }

    call(inputs: HistoryInputs): tf.Tensor3D {
        const [ B, T, C ] = this.config.batchInputShape! as number[];
        const { startingLength } = this.config;
        return tf.tidy(() => {
            const cutoffMask = tf.range(0, T).expandDims().tile([B, 1]).less(inputs[1].expandDims(-1).tile([1, T]).add(startingLength)).cast('int32'); // (B, T)
            const cutoffKeep = tf.concat([inputs[0].expandDims(1), inputs[2]], 1).slice([0, 0, 0], [B, T, C]); // (B, T, C)
            const nextHistory = cutoffKeep.div(cutoffMask.expandDims(1).tile([1, C, 1]).transpose([0, 2, 1])) as tf.Tensor3D; // (B, T, C)
            return nextHistory;
        })
    }

    computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
        return this.config.batchInputShape!;
    }

    static get className() {
        return 'History';
    }
}

tf.serialization.registerClass(History);