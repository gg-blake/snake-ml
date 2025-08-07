import * as tf from "@tensorflow/tfjs";
import { LayerArgs } from "@tensorflow/tfjs-layers/dist/engine/topology";
import "@tensorflow/tfjs-backend-webgl";

export interface FitnessGraphParams extends tf.serialization.ConfigDict {
    a: number;
    b: number;
    c: number;
    min: number;
    max: number;
}

export interface Config extends tf.serialization.ConfigDict {
    stepSize: number;
    ttl: number;
    startingLength: number;
    boundingBoxLength: number;
    units: number;
    fitnessGraphParams: FitnessGraphParams;
}

export type TensorOrArray<T, R extends tf.Rank> = T extends
    | tf.Variable
    | tf.Tensor
    ? T extends tf.Variable
        ? tf.Variable<R>
        : tf.Tensor<R>
    : never;
export type Position<T extends tf.Tensor | tf.Variable> = TensorOrArray<
    T,
    tf.Rank.R2
>;
export type Direction<T extends tf.Tensor | tf.Variable> = TensorOrArray<
    T,
    tf.Rank.R3
>;
export type History<T extends tf.Tensor | tf.Variable> = TensorOrArray<
    T,
    tf.Rank.R3
>;
export type Active<T extends tf.Tensor | tf.Variable> = TensorOrArray<
    T,
    tf.Rank.R1
>;
export type Fitness<T extends tf.Tensor | tf.Variable> = TensorOrArray<
    T,
    tf.Rank.R1
>;
export type TargetPosition<T extends tf.Tensor | tf.Variable> = TensorOrArray<
    T,
    tf.Rank.R2
>;
export type TargetIndices<T extends tf.Tensor | tf.Variable> = TensorOrArray<
    T,
    tf.Rank.R1
>;
export type FeedForwardInputs<T extends tf.Tensor | tf.Variable> =
    AugmentOutputs<T>;
export type FeedForwardOutputs<T> = TensorOrArray<T, tf.Rank.R2>;
export type MovementInputs<T extends tf.Tensor | tf.Variable> = [
    Position<T>, // Position
    Direction<T>, // Direction
    FeedForwardOutputs<T>, // Controls
    Active<tf.Tensor>, // Active
];
export type MovementOutputs<T extends tf.Tensor | tf.Variable> = [
    Position<T>, // Position
    Direction<T>, // Direction
];
export type AugmentInputs<T extends tf.Tensor | tf.Variable> = [
    Position<T>, // Position (B, C) [B, 0, C]
    Direction<T>, // Direction (B, C, C) [B, C, C]
    TargetPosition<T>, // Target (B, C) [B, C+1, C]
    History<T>, // History (B, T, C) [B, T, C]
    TargetIndices<T>, // Target Indices (B) [B]
];
export type AugmentOutputs<T extends tf.Tensor | tf.Variable> = TensorOrArray<
    T,
    tf.Rank.R3
>;
export type HistoryInputs<T extends tf.Tensor | tf.Variable> = [
    Position<T>, // Position
    TargetIndices<T>, // Target Index
    History<T>, // History
];
export type HistoryOutputs<T extends tf.Tensor | tf.Variable> = History<T>;
export type LogicInputs<T extends tf.Tensor | tf.Variable> = [
    Position<T>, // Position
    Direction<T>, // Direction
    TargetPosition<T>, // Target Position
    TargetIndices<T>, // Target Index
    Fitness<T>, // Fitness
    Active<T>, // Active
    AugmentOutputs<T>, // Input Norms Layer Output
    Position<T>, // Previous Position
];
export type LogicOutputs<T extends tf.Tensor | tf.Variable> = [
    Fitness<T>, // Fitness
    TargetIndices<T>, // Target Index
    Active<T>, // Active
];

export type LayerInputs<T extends tf.Tensor | tf.Variable> =
    | MovementInputs<T>
    | LogicInputs<T>
    | FeedForwardInputs<T>
    | HistoryInputs<T>
    | AugmentInputs<T>;
export type LayerOutputs<T extends tf.Tensor | tf.Variable> =
    | MovementOutputs<T>
    | LogicOutputs<T>
    | FeedForwardOutputs<T>
    | HistoryOutputs<T>
    | AugmentOutputs<T>;

export type LayerCallbackConfig<
    T extends tf.Tensor | tf.Variable,
    I extends LayerInputs<T>,
    O extends LayerOutputs<T>,
> = (inputs: I, config: LayerArgs & Config) => O;
export type LayerCallback<
    T extends tf.Tensor | tf.Variable,
    I extends LayerInputs<T>,
    O extends LayerOutputs<T>,
> = (inputs: I) => O;

const layerFn = <
    T extends tf.Tensor | tf.Variable,
    I extends LayerInputs<T>,
    O extends LayerOutputs<T>,
>(
    config: LayerArgs & Config,
    callback: LayerCallbackConfig<T, I, O>,
) => {
    const wrapper = (inputs: I) => tf.tidy(() => callback(inputs, config));
    return wrapper;
};

interface ModelIO<T> {
    position: TensorOrArray<T, tf.Rank.R2>; // Position of snake heads
    direction: TensorOrArray<T, tf.Rank.R3>; // Rotation matrix of all snakes
    fitness: TensorOrArray<T, tf.Rank.R1>; // Current fitness score of all snakes
    target: TensorOrArray<T, tf.Rank.R2>; // Stores the positions of active food
    targetIndices: TensorOrArray<T, tf.Rank.R1>; // Stores the snake current score
    history: TensorOrArray<T, tf.Rank.R3>; // Stores all snake parts positions
    active: TensorOrArray<T, tf.Rank.R1>; // Boolean state indicates if snake is alive
}

export { layerFn };
export type { ModelIO };