import InputNorm from "./layers/inputnorm";
import Movement from "./layers/movement";
import Logic from "./layers/logic";
import History from "./layers/history";
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import GameLayer, { GameLayerConfig, IntermediateLayer } from './layers/gamelayer';
import { settings } from "../settings";
import FeedForward from "./layers/sequential";
import { LayerArgs } from "@tensorflow/tfjs-layers/dist/engine/topology";

export default function getModel(config: GameLayerConfig): tf.LayersModel {
    const B = config.B;
    const T = config.T;
    const C = config.C;
    const position = tf.input({ batchShape: [B, C], dtype: 'float32' });
    const direction = tf.input({ batchShape: [B, C, C], dtype: 'float32' });
    const fitness = tf.input({ batchShape: [B], dtype: 'float32' });
    const target = tf.input({ batchShape: [B, C], dtype: 'float32' });
    const targetIndex = tf.input({ batchShape: [B], dtype: "int32" });
    const inputHistory = tf.input({ batchShape: [B, T, C], dtype: 'float32' });
    const active = tf.input({ batchShape: [B], name: "active", dtype: "int32" });
    const norm = new InputNorm({ batchInputShape: [B, T, C], dtype: 'float32' }, settings.model).apply([position, direction, target, inputHistory]) as tf.SymbolicTensor;
    const ffwd = new FeedForward({ batchInputShape: [B, T, C], dtype: 'float32' }, settings.model).apply(norm) as [tf.SymbolicTensor, tf.SymbolicTensor];
    const movement = new Movement({ batchInputShape: [B, T, C], dtype: 'float32' }, settings.model).apply([position, direction, ffwd[0], ffwd[1], active]) as [tf.SymbolicTensor, tf.SymbolicTensor];
    const logic = new Logic({ batchInputShape: [B, T, C], dtype: 'float32' }, settings.model).apply([movement[0], movement[1], target, targetIndex, fitness, active, norm]) as [tf.SymbolicTensor, tf.SymbolicTensor, tf.SymbolicTensor, tf.SymbolicTensor];
    const outputHistory = new History({ batchInputShape: [B, T, C], dtype: 'float32' }, settings.model).apply([movement[0], logic[1], inputHistory]) as tf.SymbolicTensor;

    const model = tf.model({
        inputs: [position, direction, fitness, target, targetIndex, inputHistory, active],
        outputs: [movement[0], movement[1], logic[0], logic[1], outputHistory, logic[2]]
    });

    return model
}
export type TensorOrArray<T, R extends tf.Rank, G> = T extends (tf.Variable | tf.Tensor) ? (T extends tf.Variable ? tf.Variable<R> : tf.Tensor<R>) : T extends number ? G : never;
export interface Data<T extends tf.Variable | tf.Tensor | number> {
    position: TensorOrArray<T, tf.Rank.R2, number[][]>;
    direction: TensorOrArray<T, tf.Rank.R3, number[][][]>;
    history: TensorOrArray<T, tf.Rank.R3, number[][][]>;
    target: TensorOrArray<T, tf.Rank.R1, number[]>;
    fitness: TensorOrArray<T, tf.Rank.R1, number[]>;
    active: TensorOrArray<T, tf.Rank.R1, number[]>;
}

export type DataValues<T extends tf.Tensor | number> = [
    Data<T>["position"],
    Data<T>["direction"],
    Data<T>["fitness"],
    Data<T>["target"],
    Data<T>["history"],
    Data<T>["active"]
];

export type ValueOf<T> = T[keyof T];

export function arraySync(a: Data<tf.Variable>): Data<number> {
    let out = {};
    for (const key of (Object.keys(a) as (keyof Data<tf.Variable>)[])) {
        // @ts-ignore
        out[key] = a[key].arraySync()
    }
    return out as Data<number>
}