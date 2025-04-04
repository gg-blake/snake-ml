import * as tf from '@tensorflow/tfjs';
import { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';
import '@tensorflow/tfjs-backend-webgl';
import GameLayer, { GameLayerConfig } from './gamelayer';

export default class FeedForward extends GameLayer {
    public wInputHidden: undefined | tf.LayerVariable;
    public bInputHidden: undefined | tf.LayerVariable;
    public wHiddenOutput: undefined | tf.LayerVariable;
    public bHiddenOutput: undefined | tf.LayerVariable;
    public seed?: number;

    constructor(config: LayerArgs, gameConfig: GameLayerConfig, seed?: number) {
        super(config, gameConfig);
        this.seed = seed;
    }

    build() {
        this.wInputHidden = this.addWeight('wInputHidden', [this.B, this.units, this.C], this.dtype, tf.initializers.randomUniform({ minval: -1, maxval: 1, seed: this.seed }))
        this.bInputHidden = this.addWeight('bInputHidden', [this.B, this.units, 1], this.dtype, tf.initializers.randomUniform({ minval: -1, maxval: 1, seed: this.seed }))
        this.wHiddenOutput = this.addWeight('wHiddenOutput', [this.B, 2 * (this.C - 1), this.units], this.dtype, tf.initializers.randomUniform({ minval: -1, maxval: 1, seed: this.seed }))
        this.bHiddenOutput = this.addWeight('bHiddenOutput', [this.B, 2 * (this.C - 1), 1], this.dtype, tf.initializers.randomUniform({ minval: -1, maxval: 1, seed: this.seed }))
    }

    call(inputs: [tf.Tensor3D]): tf.Tensor2D {
        if (!this.wInputHidden) throw new Error("Weights not initialized")
        if (!this.wHiddenOutput) throw new Error("Weights not initialized")
        
        return tf.tidy(() => {
            const hidden = tf.matMul(this.wInputHidden!.read(), inputs[0], false, false).add(this.bInputHidden!.read()).clipByValue(-1, 1);
            const logits = tf.matMul(this.wHiddenOutput!.read(), hidden, false, false).add(this.bHiddenOutput!.read()).clipByValue(-1, 1);
            /*const temp = 1;
            const probs = tf.softmax<tf.Tensor2D>(logits.squeeze([2]).div(temp) as tf.Tensor2D, 1); */// (B, (((C-1)*C)/2))
            return logits.squeeze([2]);
        });
    }

    computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
        //return inputShape;
        return [this.B, 2 * (this.C - 1)];
    }
    
    static get className() {
        return 'InputNorm';
    }
}



tf.serialization.registerClass(FeedForward);