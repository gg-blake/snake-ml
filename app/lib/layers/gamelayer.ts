import * as tf from '@tensorflow/tfjs';
import { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';
import '@tensorflow/tfjs-backend-webgl';

export interface FitnessGraphParams extends tf.serialization.ConfigDict {
    a: number;
    b: number;
    c: number;
    min: number;
    max: number;
}

export interface Config extends tf.serialization.ConfigDict {
    ttl: number;
    startingLength: number;
    boundingBoxLength: number;
    units: number;
    fitnessGraphParams: FitnessGraphParams;
}

export type TensorOrArray<T, R extends tf.Rank, G> = T extends (tf.Variable | tf.Tensor) ? (T extends tf.Variable ? tf.Variable<R> : tf.Tensor<R>) : T extends number ? G : never;
export type Position<T extends tf.Tensor | tf.Variable | number> = TensorOrArray<T, tf.Rank.R2, number[][]>;
export type Direction<T extends tf.Tensor | tf.Variable | number> = TensorOrArray<T, tf.Rank.R3, number[][][]>;
export type History<T extends tf.Tensor | tf.Variable | number> = TensorOrArray<T, tf.Rank.R3, number[][][]>;
export type Active<T extends tf.Tensor | tf.Variable | number> = TensorOrArray<T, tf.Rank.R1, number[][]>;
export type Fitness<T extends tf.Tensor | tf.Variable | number> = TensorOrArray<T, tf.Rank.R1, number[][]>;
export type TargetPosition<T extends tf.Tensor | tf.Variable | number> = TensorOrArray<T, tf.Rank.R2, number[][]>;
export type TargetIndices<T extends tf.Tensor | tf.Variable | number> = TensorOrArray<T, tf.Rank.R1, number[]>;


export default abstract class GameLayer extends tf.layers.Layer {
    config: LayerArgs & Config;

    constructor(config: LayerArgs & Config, name?: string) {
        super({
            ...config,
            name: name
        });
        
        if (!config.batchInputShape) {
            throw new Error("GameLayer input shape is not specified");
        }

        this.config = config;
        
    }

    getConfig(): LayerArgs & Config {
        return this.config;
    }
}

