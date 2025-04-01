import * as tf from '@tensorflow/tfjs';
import { NamedTensor } from '@tensorflow/tfjs-core/dist/tensor_types';
import GameLayer, { GameLayerConfig } from './layers/gamelayer';
import { Data, DataValues } from './model';
import getModel from './model';
import { NEATRenderer, Renderer } from './renderer';
import { generatePlaneIndices } from './util';
import InputNorm from './layers/inputnorm';
import Logic from './layers/logic';

interface NEATVariableGradients extends tf.NamedTensorMap {
    weights: tf.Tensor;
}

export interface NEATConfig extends tf.serialization.ConfigDict {
    mutationRate: number;
    mutationFactor: number;
}

export interface Trainer {
    stepCount: number;
    config: GameLayerConfig;
    target: tf.Variable<tf.Rank.R2>; // Stores the fixed number of generated targets for population to be tested on
    state: Data<tf.Variable>; // Tracks the current stats of the performing population

    step(model: tf.LayersModel | tf.LayersModel[]): void;

    resetState(): void;

    set setState(values: DataValues<tf.Tensor> | Data<tf.Tensor> | Data<tf.Variable>);
}

export interface NEATI extends Trainer {
    mutationRate: number; // How frequent a mutation occurs in the population's gene pool
    mutationFactor: number; // How much each mutation affect the existing weights

    set setState(values: DataValues<tf.Tensor>);

    evolve(model: tf.LayersModel): void;
}

export default class NEAT implements NEATI {
    epochCount: number;
    stepCount: number;
    config: GameLayerConfig;
    mutationRate: number; // How frequent a mutation occurs in the population's gene pool
    mutationFactor: number; // How much each mutation affect the existing weights
    fitnessDelta: tf.Variable<tf.Rank.R1>;
    target: tf.Variable<tf.Rank.R2>; // Stores the fixed number of generated targets for population to be tested on
    state: Data<tf.Variable>; // Tracks the current stats of the performing population
    planeIndices: tf.Tensor2D;
    timeToRestart: tf.Variable<tf.Rank.R0>;
    batchMaxFitness: tf.Variable<tf.Rank.R0>;

    constructor(gameConfig: GameLayerConfig, gameOptimizerConfig: NEATConfig) {
        this.epochCount = 0;
        this.stepCount = 0;
        this.config = gameConfig;
        this.mutationRate = gameOptimizerConfig.mutationRate;
        this.mutationFactor = gameOptimizerConfig.mutationFactor;
        this.fitnessDelta = tf.variable(tf.zeros([gameConfig.B]));
        this.target = tf.variable(tf.randomUniform([this.config.T, this.config.C], -this.config.boundingBoxLength, this.config.boundingBoxLength, 'float32'))
        this.state = {
            position: tf.variable(tf.zeros([this.config.B, this.config.C], 'float32')),
            direction: tf.variable(tf.eye(this.config.C, ...[, ,], 'float32').expandDims(0).tile([this.config.B, 1, 1])),
            fitness: tf.variable(tf.ones([this.config.B], 'float32').mul(this.config.TTL)),
            target: tf.variable(tf.zeros([this.config.B], 'int32')),
            history: tf.variable(tf.zeros([this.config.B, this.config.T, this.config.C], 'float32')),
            active: tf.variable(tf.ones([this.config.B], 'int32'))
        }
        this.planeIndices = generatePlaneIndices(this.config.C);
        this.timeToRestart = tf.variable(tf.tensor(this.config.TTL));
        this.batchMaxFitness = tf.variable(tf.tensor(this.config.TTL));
    }

    step(model: tf.LayersModel, renderer?: NEATRenderer): void {
        tf.tidy(() => {
            if (this.timeToRestart.arraySync() < 0) {
                if (this.state.target.greater(0).any().arraySync() == 1) {
                    this.target.assign(tf.concat([tf.randomUniform([1, this.config.C]), this.target.slice([1, 0], [this.config.T - 1, this.config.C])], 0));
                }

                if (this.config.B > 1) {
                    this.evolve(model);
                }
                
                this.resetState();
                this.stepCount = 0;
                this.epochCount++;
                this.timeToRestart.assign(tf.tensor(this.config.TTL));
                this.batchMaxFitness.assign(tf.tensor(this.config.TTL));
                
                return;
            }

            const prevState = this.state.fitness.clone();
            const prevBatchMaxFitness = this.state.fitness.max(-1);
            this.setState = model.predictOnBatch([
                this.state.position,
                this.state.direction,
                this.state.fitness,
                this.target.gather(this.state.target),
                this.state.target,
                this.state.history,
                this.state.active
            ]) as DataValues<tf.Tensor>;
            this.fitnessDelta.assign(this.state.fitness.sub(prevState));
            this.batchMaxFitness.assign(tf.max(tf.stack([this.state.fitness.mul(this.state.active).max(-1), this.batchMaxFitness])))
            const nextBatchMaxFitness = this.state.fitness.max(-1);
            const batchMaxFitnessDelta = this.state.fitness.mul(this.state.active).max(-1).sub(this.batchMaxFitness);
            this.stepCount++;

            this.timeToRestart.assign(this.timeToRestart.add(batchMaxFitnessDelta.clipByValue(-1, 0)) as tf.Scalar);
        })

        if (renderer == null) {
            return;
        }

        renderer.renderStep(this);
    }

    resetState() {
        this.fitnessDelta.assign(tf.zeros([this.config.B]));
        this.setState = [
            tf.zeros([this.config.B, this.config.C], 'float32'),
            tf.eye(this.config.C, ...[, ,], 'float32').expandDims(0).tile([this.config.B, 1, 1]),
            tf.ones([this.config.B], 'float32').mul(this.config.TTL),
            tf.zeros([this.config.B], 'int32'),
            tf.zeros([this.config.B, this.config.T, this.config.C], 'float32'),
            tf.ones([this.config.B], 'int32')
        ];
    }

    set setState(values: DataValues<tf.Tensor>) {
        this.state.position.assign(values[0]);
        this.state.direction.assign(values[1]);
        this.state.fitness.assign(values[2]);
        this.state.target.assign(values[3]);
        this.state.history.assign(values[4]);
        this.state.active.assign(values[5]);
    }

    evolve(model: tf.LayersModel) {
        tf.tidy(() => {
            let probs = this.state.fitness.sub(this.state.fitness.sum().div(this.config.B)).clipByValue(0, Infinity).softmax() as tf.Tensor1D;
            const indicesA = tf.multinomial(probs, this.config.B);
            const indicesB = tf.multinomial(probs, this.config.B);
            const weights = model.getWeights(true);
            // Selector operator
            const weightsA = weights.map((layer: tf.Tensor) => layer.gather(indicesA));
            const weightsB = weights.map((layer: tf.Tensor) => layer.gather(indicesB));
            let newWeights = [];
            for (let layerI = 0; layerI < weights.length; layerI++) {
                // Crossover operator
                const mask = tf.randomUniformInt(weights[layerI].shape, 0, 1).cast('float32');
                const weightsFromA = weightsA[layerI].mul(mask); // Randomly select genes from A
                const weightsFromB = weightsB[layerI].mul(mask.sub(1).abs()); // Wherever genes aren't inherited from A are inherited from B
                const newLayer = tf.add(weightsFromA, weightsFromB);
                // Mutation operator
                const mutationMask = tf.randomUniform(weights[layerI].shape, 0, 1).lessEqual(this.mutationRate).cast('float32');
                const mutationValue = tf.randomUniform(weights[layerI].shape, -1, 1).mul(this.mutationFactor).mul(mutationMask);
                const newLayerMutated = tf.clipByValue(newLayer.add(mutationValue), -1, 1); // Keep weights within interval [-1, 1]
                newWeights.push(newLayerMutated);
            }

            model.setWeights(newWeights);
        })
    }
}