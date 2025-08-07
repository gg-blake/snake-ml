import * as tf from "@tensorflow/tfjs";
import { LayerArgs } from "@tensorflow/tfjs-layers/dist/engine/topology";
import "@tensorflow/tfjs-backend-webgl";
import { Config, HistoryInputs, HistoryOutputs, LayerCallbackConfig } from "./types";

const history: LayerCallbackConfig<tf.Tensor, HistoryInputs<tf.Tensor>, HistoryOutputs<tf.Tensor>> = (
    inputs,
    config,
): tf.Tensor3D => {
    const [B, T, C] = config.batchInputShape! as number[];
    const { startingLength } = config;
    return tf.tidy(() => {
        const cutoffMask = tf
            .range(0, T)
            .expandDims()
            .tile([B, 1])
            .less(inputs[1].expandDims(-1).tile([1, T]).add(startingLength))
            .cast("int32"); // (B, T)
        const cutoffKeep = tf
            .concat([inputs[0].expandDims(1), inputs[2]], 1)
            .slice([0, 0, 0], [B, T, C]); // (B, T, C)
        const nextHistory = cutoffKeep.div(
            cutoffMask.expandDims(1).tile([1, C, 1]).transpose([0, 2, 1]),
        ) as tf.Tensor3D; // (B, T, C)
        return nextHistory;
    });
};

export { history };
