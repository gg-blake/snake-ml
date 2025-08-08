import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import { Config } from "./types";

export default class FeedForward extends tf.layers.Layer {
    public wInputHidden: undefined | tf.LayerVariable;
    public mInputHidden: undefined | tf.LayerVariable;
    public wHiddenOutput: undefined | tf.LayerVariable;
    public mHiddenOutput: undefined | tf.LayerVariable;
    public seed?: number;
    config: Config;

    constructor(config: Config, seed?: number) {
        super(config);
        this.config = config;
        this.seed = seed;
    }

    build() {
        const [B, T, C] = this.config.batchInputShape! as number[];
        const { units } = this.config;
        this.wInputHidden = this.addWeight(
            "wInputHidden",
            [B, units, 2 * (C - 1) + 1],
            this.dtype,
            tf.initializers.randomUniform({
                minval: -1,
                maxval: 1,
                seed: this.seed,
            }),
            ...[,],
            true,
        );
        this.wHiddenOutput = this.addWeight(
            "wHiddenOutput",
            [B, 2 * (C - 1), units],
            this.dtype,
            tf.initializers.randomUniform({
                minval: -1,
                maxval: 1,
                seed: this.seed,
            }),
            ...[,],
            true,
        );
    }

    call(inputs: [tf.Tensor3D]): tf.Tensor2D {
        return tf.tidy(() => {
            const ih = this.wInputHidden!.read();
            const ho = this.wHiddenOutput!.read();
            const hidden = tf
                .matMul(ih, inputs[0], false, false)
                .clipByValue(-1, 1);
            const logits = tf
                .matMul(ho, hidden, false, false)
                .clipByValue(-1, 1)
                .squeeze([2]);
            return logits as tf.Tensor2D;
        });
    }

    computeOutputShape(inputShape: tf.Shape): tf.Shape {
        const [B, T, C] = this.config.batchInputShape! as number[];
        return [B, 2 * (C - 1)];
    }

    static get className() {
        return "FeedForward";
    }
}

tf.serialization.registerClass(FeedForward);
