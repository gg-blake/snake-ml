import InputNorm from "./layers/inputnorm";
import Movement from "./layers/movement";
import Logic from "./layers/logic";
import History from "./layers/history";
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import { Settings } from "../settings";
import FeedForward from "./layers/sequential";
import * as GL from "./layers/gamelayer";

export default function getModel(settings: Settings["model"], seed?: number): tf.LayersModel {
    const [ B, T, C ] = settings.batchInputShape! as number[];
    const position = tf.input({ batchShape: [B, C], dtype: 'float32' });
    const direction = tf.input({ batchShape: [B, C, C], dtype: 'float32' });
    const fitness = tf.input({ batchShape: [B], dtype: 'float32' });
    const target = tf.input({ batchShape: [B, C], dtype: 'float32' });
    const targetIndex = tf.input({ batchShape: [B], dtype: "int32" });
    const inputHistory = tf.input({ batchShape: [B, T, C], dtype: 'float32' });
    const active = tf.input({ batchShape: [B], dtype: "int32" });
    const norm = new InputNorm(settings).apply([position, direction, target, inputHistory]) as tf.SymbolicTensor;
    const ffwd = new FeedForward(settings, seed).apply(norm) as tf.SymbolicTensor;
    const movement = new Movement(settings).apply([position, direction, ffwd, active]) as [tf.SymbolicTensor, tf.SymbolicTensor];
    const logic = new Logic(settings).apply([movement[0], movement[1], target, targetIndex, fitness, active, norm, position]) as [tf.SymbolicTensor, tf.SymbolicTensor, tf.SymbolicTensor, tf.SymbolicTensor];
    const outputHistory = new History(settings).apply([movement[0], logic[1], inputHistory]) as tf.SymbolicTensor;

    const model = tf.model({
        inputs: [position, direction, fitness, target, targetIndex, inputHistory, active],
        outputs: [movement[0], movement[1], logic[0], logic[1], outputHistory, logic[2]]
    });

    return model
}
export interface Data<T extends tf.Variable | tf.Tensor | number> {
    position: GL.Position<T>;
    direction: GL.Direction<T>;
    history: GL.History<T>;
    target: GL.TargetIndices<T>;
    fitness: GL.Fitness<T>;
    active: GL.Active<T>;
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