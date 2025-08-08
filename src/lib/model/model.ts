import { LayerArgs } from "@tensorflow/tfjs-layers/dist/engine/topology";
import * as tf from "@tensorflow/tfjs";
import FeedForward from "./utils/sequential";
import { movement } from "./utils/movement";
import { logic } from "./utils/logic";
import { augment } from "./utils/augment";
import { history } from "./utils/history";
import {
    ModelIO,
    LayerCallback,
    HistoryInputs,
    HistoryOutputs,
    MovementInputs,
    MovementOutputs,
    AugmentInputs,
    AugmentOutputs,
    LogicInputs,
    LogicOutputs,
    Config,
    layerFn,
} from "./utils/types";
import Logging from "../logger";

class ModelInput extends Logging implements ModelIO<tf.Variable> {
    position: tf.Variable<tf.Rank.R2>; // Position of snake heads
    direction: tf.Variable<tf.Rank.R3>; // Rotation matrix of all snakes
    fitness: tf.Variable<tf.Rank.R1>; // Current fitness score of all snakes
    target: tf.Variable<tf.Rank.R2>; // Stores the positions of active food
    targetIndices: tf.Variable<tf.Rank.R1>; // Stores the snake current score
    history: tf.Variable<tf.Rank.R3>; // Stores all snake parts positions
    active: tf.Variable<tf.Rank.R1>; // Boolean state indicates if snake is alive

    constructor(shape: [number, number, number], debug?: boolean) {
        super(debug);
        const [B, T, C] = shape;
        this.position = tf.variable(tf.zeros([B, C]));
        this.direction = tf.variable(
            tf
                .eye(C, ...[, ,], "float32")
                .expandDims(0)
                .tile([B, 1, 1]),
        );
        this.fitness = tf.variable(tf.ones([B]));
        this.target = tf.variable(tf.zeros([T, C]));
        this.targetIndices = tf.variable(tf.zeros([B], "int32"));
        this.history = tf.variable(tf.zeros([B, T, C]));
        this.active = tf.variable(tf.ones([B], "int32"));
    }

    assign(state: ModelIO<tf.Tensor>) {
        Object.entries(state).map((entry, index) => {
            const key = entry[0] as keyof ModelIO<tf.Tensor>;

            // @ts-ignore
            this[key].assign(state[key]);
        });
    }
    
    unmount(verbose?: boolean) {
        const propertyNames = Object.getOwnPropertyNames(ModelInput.prototype).filter(
            (name) =>
                name !== "constructor" &&
                typeof (this as any)[name] !== "function"
        );
        for (const name of propertyNames) {
            (this as any)[name].dispose();
        }
    }
}

class ModelState extends ModelInput {
    model: Model; // Batch snake model contains all snake population weights
    updateCount: number;

    constructor(model: Model) {
        super(model.shape);
        this.model = model;
        this.updateCount = 0;
    }

    update() {
        const logits = this.model.call(this);
        this.assign(logits);
        this.updateCount++;
    }
    
    
}

type History = LayerCallback<
    tf.Tensor,
    HistoryInputs<tf.Tensor>,
    HistoryOutputs<tf.Tensor>
>;

type Movement = LayerCallback<
    tf.Tensor,
    MovementInputs<tf.Tensor>,
    MovementOutputs<tf.Tensor>
>;

type Logic = LayerCallback<
    tf.Tensor,
    LogicInputs<tf.Tensor>,
    LogicOutputs<tf.Tensor>
>;

type Augment = LayerCallback<
    tf.Tensor,
    AugmentInputs<tf.Tensor>,
    AugmentOutputs<tf.Tensor>
>;

class Model extends Logging {
    _ffwd: tf.LayersModel;
    _movement: Movement;
    _logic: Logic;
    _augment: Augment;
    _history: History;
    _config: Config;
    batchSize: number; // Number of snakes
    timeSize: number; // Max length of snakes (position history)
    channelSize: number; // Number of spatial dimensions

    constructor(config: Config, debug?: boolean) {
        super(debug);

        // Initialize properties
        this._config = config;
        this.batchSize = this._config.batchInputShape![0] as number;
        this.timeSize = this._config.batchInputShape![1] as number;
        this.channelSize = this._config.batchInputShape![2] as number;
        this._movement = layerFn(this._config, movement);
        this._history = layerFn(this._config, history);
        this._augment = layerFn(this._config, augment);
        this._logic = layerFn(this._config, logic);

        // Build weights
        this._ffwd = null!;
        this.build();
    }

    build() {
        const ffwd = new FeedForward(this._config);
        const [B, T, C] = this.shape;
        const input = tf.input({ batchShape: [B, 2 * (C - 1) + 1, 1] });
        const output = ffwd.apply(input) as tf.SymbolicTensor;
        this._ffwd = tf.model({
            inputs: [input],
            outputs: [output],
        });
    }

    get shape(): [number, number, number] {
        return [this.batchSize, this.timeSize, this.channelSize];
    }

    call(input: ModelIO<tf.Variable> | ModelIO<tf.Tensor>): ModelIO<tf.Tensor> {
        return tf.tidy(() => {
            const preFFWD = this._augment([
                input.position,
                input.direction,
                input.target,
                input.history,
                input.targetIndices,
            ]);

            const outFFWD = this._ffwd.predictOnBatch(preFFWD) as tf.Tensor2D;

            const [nextPosition, nextDirection] = this._movement([
                input.position,
                input.direction,
                outFFWD,
                input.active,
            ]);

            const [nextFitness, nextTargetIndices, isAliveMask] = this._logic([
                nextPosition,
                nextDirection,
                input.target,
                input.targetIndices,
                input.fitness,
                input.active,
                preFFWD,
                input.position,
            ]);
            const nextHistory = this._history([
                nextPosition,
                nextTargetIndices,
                input.history,
            ]);

            return {
                position: tf.keep(nextPosition),
                direction: tf.keep(nextDirection),
                fitness: tf.keep(nextFitness),
                target: input.target,
                targetIndices: tf.keep(nextTargetIndices),
                history: tf.keep(nextHistory),
                active: tf.keep(isAliveMask),
            };
        });
    }

    getWeights(trainable?: boolean): tf.Tensor[] {
        return this._ffwd.getWeights(trainable);
    }

    setWeights(weights: tf.Tensor[]): void {
        this._ffwd.setWeights(weights);
    }

    unmount(verbose?: boolean) {
        this._ffwd.dispose();
        if (!verbose) return;
        this.log("unmounted");
    }
}

export { ModelState, Model, ModelInput };
