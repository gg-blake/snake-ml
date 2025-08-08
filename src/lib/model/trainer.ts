import * as tf from "@tensorflow/tfjs";
import { ModelState, Model } from "./model";
import { Fitness, Config, Mountable } from "./utils/types";
import { LayerArgs } from "@tensorflow/tfjs-layers/dist/engine/topology";
import Logging from "../logger";

interface TrainingConfig {
    mutationRate: number;
    mutationFactor: number;
    timeToLive: number;
}

class TrainerState {
    updateCounts: number[];
    bestFitnesses: number[];
    epochCount: number;

    constructor() {
        this.updateCounts = [];
        this.bestFitnesses = [];
        this.epochCount = 0;
    }

    record(state: ModelState) {
        this.epochCount++;
    }
}

class Trainer extends Logging implements Mountable {
    modelState: ModelState;
    trainerState: TrainerState;
    config: TrainingConfig;

    constructor(
        modelState: ModelState,
        trainerState: TrainerState,
        trainingConfig: TrainingConfig,
        debug?: boolean,
    ) {
        super(debug);
        this.modelState = modelState;
        this.trainerState = trainerState;
        this.config = trainingConfig;

        // Initialize fitness to be the timeToLive
        this.modelState.fitness.assign(
            this.modelState.fitness.mul(this.config.timeToLive),
        );

        // Initialize target positions to random coords within the bounding box
        const [B, T, C] = this.modelState.model.shape;
        const { boundingBoxLength } = this.modelState.model._config;
        this.modelState.target.assign(
            tf.randomUniform(
                [T, C],
                -boundingBoxLength,
                boundingBoxLength,
                "float32",
            ),
        );
    }

    step() {
        if (this.isDead()) {
            console.log("All snakes dead");
            tf.tidy(() => this.updateWeights());
            this.resetState();
        }

        this.modelState.update();
    }

    resetState() {
        this.trainerState.record(this.modelState); // Record/calculate stats
        const model = this.modelState.model;
        this.modelState = new ModelState(model);
    }

    updateWeights() {
        let newWeights = [];
        const batchSize = this.modelState.model.batchSize;
        const fitness = this.modelState.fitness;
        const weights = this.modelState.model.getWeights(true);
        // Natural selection
        const [weightsA, weightsB] = selectParents(batchSize, fitness, weights);
        for (let layerIdx = 0; layerIdx < weights.length; layerIdx++) {
            // Crossover operator
            const originalLayer = weights[layerIdx];
            const layerA = weightsA[layerIdx];
            const layerB = weightsB[layerIdx];
            const crossoverLayer = crossoverParentLayer(
                originalLayer,
                layerA,
                layerB,
            );

            // Mutation operator
            const mutatedLayer = mutateParentLayer(
                originalLayer,
                crossoverLayer,
                this.config.mutationRate,
                this.config.mutationFactor,
            );
            newWeights.push(mutatedLayer);
        }

        this.modelState.model.setWeights(newWeights);
    }

    isDead(): boolean {
        return this.modelState.active.greater(0).any().arraySync() != 1;
    }
    
    unmount(verbose?: boolean) {
        this.modelState.model.unmount();
        if (!verbose) return;
        this.log("unmounted");
    }
}

function mutateParentLayer(
    originalLayer: tf.Tensor,
    newLayer: tf.Tensor,
    mutationRate: number,
    mutationFactor: number,
): tf.Tensor {
    const mutationMask = tf
        .randomUniform(originalLayer.shape, 0, 1)
        .lessEqual(mutationRate)
        .cast("float32");
    const mutationValue = tf
        .randomUniform(originalLayer.shape, -mutationFactor, mutationFactor)
        .mul(mutationMask);
    const newLayerMutated = tf.clipByValue(newLayer.add(mutationValue), -1, 1); // Keep weights within interval [-1, 1]
    return newLayerMutated;
}

function crossoverParentLayer(
    originalLayer: tf.Tensor,
    layerA: tf.Tensor,
    layerB: tf.Tensor,
): tf.Tensor {
    const mask = tf.randomUniformInt(originalLayer.shape, 0, 1).cast("float32");
    const weightsFromA = layerA.mul(mask); // Randomly select genes from A
    const weightsFromB = layerB.mul(mask.sub(1).abs()); // Wherever genes aren't inherited from A are inherited from B
    const newLayer = tf.add(weightsFromA, weightsFromB);
    return newLayer;
}

// Select the most fit parents based on high fitness scores
function selectParents<T extends tf.Tensor | tf.Variable>(
    batchSize: number,
    fitness: Fitness<T>,
    weights: tf.Tensor[],
): [tf.Tensor[], tf.Tensor[]] {
    let probs = fitness
        .sub(fitness.sum().div(batchSize))
        .clipByValue(0, Infinity)
        .softmax() as tf.Tensor1D;
    const indicesA = tf.multinomial(probs, batchSize);
    const indicesB = tf.multinomial(probs, batchSize);

    const weightsA = weights.map((layer: tf.Tensor) => layer.gather(indicesA));
    const weightsB = weights.map((layer: tf.Tensor) => layer.gather(indicesB));

    return [weightsA, weightsB];
}

// vvvv Production implementation below vvvv
var modelConfig: Config = {
    stepSize: 1,
    ttl: 200,
    batchInputShape: [100, 10, 3],
    startingLength: 5,
    boundingBoxLength: 30,
    units: 24,
    fitnessGraphParams: {
        a: 10,
        b: 1.5,
        c: 4,
        min: -1,
        max: 1,
    },
    dtype: "float32",
};

var trainingConfig: TrainingConfig = {
    mutationFactor: 0.1,
    mutationRate: 0,
    timeToLive: 20,
};

function getTrainer(debug?: boolean): Trainer {
    const model = new Model(modelConfig, debug);
    const modelState = new ModelState(model);
    const trainingState = new TrainerState();
    const trainer = new Trainer(
        modelState,
        trainingState,
        trainingConfig,
        debug,
    );
    return trainer;
}

export { Trainer, TrainerState, trainingConfig, modelConfig };
export type { TrainingConfig };
export default getTrainer;
