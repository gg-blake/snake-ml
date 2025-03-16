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

export interface GameLayerConfig extends tf.serialization.ConfigDict {
    B: number;
    T: number;
    C: number;
    TTL: number;
    startingLength: number;
    boundingBoxLength: number;
    units: number;
    fitnessGraphParams: FitnessGraphParams;
}

export type IntermediateLayerInputSignal = Array<tf.Tensor<any> | tf.TensorLike>;
export type IntermediateLayerOutputSignal = IntermediateLayerInputSignal | tf.Tensor<any> | tf.TensorLike;
export type IntermediateLayer<I extends IntermediateLayerInputSignal, O extends IntermediateLayerOutputSignal> = (B: number, T: number, C: number, ...args: I) => O;

export default class GameLayer extends tf.layers.Layer {
    B: number;
    T: number;
    C: number;
    TTL: number;
    startingLength: number;
    boundingBoxLength: number;
    units: number;
    fitnessGraphParams: FitnessGraphParams;

    constructor(config: LayerArgs, gameConfig: GameLayerConfig) {
        super(config);
        if (!gameConfig) {
            throw new Error("Game config not provided");
        }
        if (!config.batchInputShape) {
            throw new Error("GameLayer input shape is not specified");
        }

        this.TTL = gameConfig.TTL;
        this.startingLength = gameConfig.startingLength;
        this.boundingBoxLength = gameConfig.boundingBoxLength;
        this.units = gameConfig.units;
        this.fitnessGraphParams = gameConfig.fitnessGraphParams;
        this.B = config.batchInputShape[0]!;
        this.T = config.batchInputShape[1]!;
        this.C = config.batchInputShape[2]!;
        
    }

    getConfig(): LayerArgs & GameLayerConfig {
        const config = super.getConfig() as LayerArgs & GameLayerConfig;
        Object.assign(config, {
            B: this.B,
            T: this.T,
            C: this.C,
            TTL: this.TTL,
            startingLength: this.startingLength,
            boundingBoxLength: this.boundingBoxLength,
            units: this.units,
            fitnessGraphParams: this.fitnessGraphParams
        });
        return config;
    }
}

