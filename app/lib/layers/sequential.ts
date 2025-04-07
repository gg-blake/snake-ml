import * as tf from '@tensorflow/tfjs';
import { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';
import '@tensorflow/tfjs-backend-webgl';
import GameLayer, * as GL from './gamelayer';

export default class FeedForward extends GameLayer {
    public wInputHidden: undefined | tf.LayerVariable;
    public mInputHidden: undefined | tf.LayerVariable;
    public wHiddenOutput: undefined | tf.LayerVariable;
    public mHiddenOutput: undefined | tf.LayerVariable;
    public seed?: number;

    constructor(config: LayerArgs & GL.Config, seed?: number) {
        super(config, FeedForward.className);
        this.seed = seed;
    }

    build() {
        const [ B, T, C ] = this.config.batchInputShape! as number[];
        const { units } = this.config;
        this.wInputHidden = this.addWeight('wInputHidden', [B, units, 2 * (C - 1) + 1], this.dtype, tf.initializers.randomUniform({ minval: -1, maxval: 1, seed: this.seed }), ...[,], true)
        this.mInputHidden = this.addWeight('mInputHidden', [B, units, 2 * (C - 1) + 1], this.dtype, tf.initializers.randomUniform({ minval: -1, maxval: 1, seed: this.seed }), ...[,], true)
        this.wHiddenOutput = this.addWeight('wHiddenOutput', [B, 2 * (C - 1), units], this.dtype, tf.initializers.randomUniform({ minval: -1, maxval: 1, seed: this.seed }), ...[,], true)
        this.mHiddenOutput = this.addWeight('mHiddenOutput', [B, 2 * (C - 1), units], this.dtype, tf.initializers.randomUniform({ minval: -1, maxval: 1, seed: this.seed }), ...[,], true)
        
    }

    call(inputs: [tf.Tensor3D]): tf.Tensor2D {
        return tf.tidy(() => {
            const hiddenMask = this.wInputHidden!.read().mul(this.mInputHidden!.read().greaterEqual(0).cast(this.dtype));
            const hidden = tf.matMul(hiddenMask, inputs[0], false, false).clipByValue(-1, 1);
            const outputMask = this.wHiddenOutput!.read().mul(this.mHiddenOutput!.read().greaterEqual(0).cast(this.dtype));
            const logits = tf.matMul(outputMask, hidden, false, false).clipByValue(-1, 1).squeeze([2]);
            return logits as tf.Tensor2D;
        });
    }

    computeOutputShape(inputShape: tf.Shape): tf.Shape {
        const [ B, T, C ] = this.config.batchInputShape! as number[];
        return [B, 2 * (C - 1)];
    }
    
    static get className() {
        return 'FeedForward';
    }
}



tf.serialization.registerClass(FeedForward);