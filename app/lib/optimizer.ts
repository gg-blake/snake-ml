import * as tf from '@tensorflow/tfjs';
import { NamedTensor } from '@tensorflow/tfjs-core/dist/tensor_types';
import GameLayer, * as GL from './layers/gamelayer';
import { Data, DataValues } from './model';
import { NEATRenderer, Renderer } from './renderer';
import { generatePlaneIndices } from './util';
import InputNorm from './layers/inputnorm';
import Logic from './layers/logic';
import { Settings } from '../settings';

interface NEATVariableGradients extends tf.NamedTensorMap {
    weights: tf.Tensor;
}

export interface NEATConfig extends tf.serialization.ConfigDict {
    mutationRate: number;
    mutationFactor: number;
}

export interface Trainer {
    stepCount: number;
    config: Settings;
    target: GL.TargetPosition<tf.Variable<tf.Rank.R2>>; // Stores the fixed number of generated targets for population to be tested on
    state: Data<tf.Variable>; // Tracks the current stats of the performing population

    step(model: tf.LayersModel | tf.LayersModel[]): void;

    resetState(): void;

    set setState(values: DataValues<tf.Tensor> | Data<tf.Tensor> | Data<tf.Variable>);
}

export interface NEATI extends Trainer {
    set setState(values: DataValues<tf.Tensor>);

    evolve(model: tf.LayersModel): void;
}

export default class NEAT implements NEATI {
    epochCount: number;
    stepCount: number;
    config: Settings;
    fitnessDelta: tf.Variable<tf.Rank.R1>;
    target: tf.Variable<tf.Rank.R2>; // Stores the fixed number of generated targets for population to be tested on
    state: Data<tf.Variable>; // Tracks the current stats of the performing population
    planeIndices: tf.Tensor2D;
    timeToRestart: tf.Variable<tf.Rank.R0>;
    batchMaxFitness: tf.Variable<tf.Rank.R0>;

    constructor(config: Settings) {
        const [ B, T, C ] = config.model.batchInputShape! as number[];
        const { ttl, boundingBoxLength } = config.model;
        this.epochCount = 0;
        this.stepCount = 0;
        this.config = config;
        this.fitnessDelta = tf.variable(tf.zeros([B]));
        this.target = tf.variable(tf.randomUniform([T, C], -boundingBoxLength, boundingBoxLength, 'float32'))
        this.state = {
            position: tf.variable(tf.zeros([B, C], 'float32')),
            direction: tf.variable(tf.eye(C, ...[, ,], 'float32').expandDims(0).tile([B, 1, 1])),
            fitness: tf.variable(tf.ones([B], 'float32').mul(ttl)),
            target: tf.variable(tf.zeros([B], 'int32')),
            history: tf.variable(tf.zeros([B, T, C], 'float32')),
            active: tf.variable(tf.ones([B], 'int32'))
        }
        this.planeIndices = generatePlaneIndices(C);
        this.timeToRestart = tf.variable(tf.tensor(ttl));
        this.batchMaxFitness = tf.variable(tf.tensor(ttl));
    }

    step(model: tf.LayersModel, renderer?: NEATRenderer): void {
        const [ B, T, C ] = this.config.model.batchInputShape! as number[];
        const { ttl } = this.config.model;
        tf.tidy(() => {
            if (this.timeToRestart.arraySync() < 0) {
                if (this.state.target.greater(0).any().arraySync() == 1) {
                    this.target.assign(tf.concat([tf.randomUniform([1, C]), this.target.slice([1, 0], [T - 1, C])], 0));
                }

                if (B > 1) {
                    this.evolve(model);
                }
                
                this.resetState();
                this.stepCount = 0;
                this.epochCount++;
                this.timeToRestart.assign(tf.tensor(ttl));
                this.batchMaxFitness.assign(tf.tensor(ttl));
                
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

            this.timeToRestart.assign(this.timeToRestart.add(batchMaxFitnessDelta.clipByValue(-1, 0).sub(0.1)) as tf.Scalar);
        })

        if (renderer == null) {
            return;
        }

        renderer.renderStep(this);
    }

    resetState() {
        const [ B, T, C ] = this.config.model.batchInputShape! as number[];
        const { ttl } = this.config.model;
        this.fitnessDelta.assign(tf.zeros([B]));
        this.setState = [
            tf.zeros([B, C], 'float32'),
            tf.eye(C, ...[, ,], 'float32').expandDims(0).tile([B, 1, 1]),
            tf.ones([B], 'float32').mul(ttl),
            tf.zeros([B], 'int32'),
            tf.zeros([B, T, C], 'float32'),
            tf.ones([B], 'int32')
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
        const [ B, T, C ] = this.config.model.batchInputShape! as number[];
        const { mutationRate, mutationFactor } = this.config.trainer;
        tf.tidy(() => {
            let probs = this.state.fitness.sub(this.state.fitness.sum().div(B)).clipByValue(0, Infinity).softmax() as tf.Tensor1D;
            const indicesA = tf.multinomial(probs, B);
            const indicesB = tf.multinomial(probs, B);
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
                const mutationMask = tf.randomUniform(weights[layerI].shape, 0, 1).lessEqual(mutationRate).cast('float32');
                const mutationValue = tf.randomUniform(weights[layerI].shape, -mutationFactor, mutationFactor).mul(mutationMask);
                const newLayerMutated = tf.clipByValue(newLayer.add(mutationValue), -1, 1); // Keep weights within interval [-1, 1]
                newWeights.push(newLayerMutated);
            }

            model.setWeights(newWeights);
        })
    }
}