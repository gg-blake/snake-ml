import * as tf from '@tensorflow/tfjs';
import { NamedTensor } from '@tensorflow/tfjs-core/dist/tensor_types';
import GameLayer, { GameLayerConfig } from './layers/gamelayer';
import { settings } from '../settings';

interface NEATVariableGradients extends tf.NamedTensorMap {
    weights: tf.Tensor;
}

export interface NEATConfig extends tf.serialization.ConfigDict {
    mutationRate: number;
}

export default class NEAT extends tf.Optimizer {
    step: number;
    B: number;
    T: number;
    C: number;
    mutationRate: number;
    fitness: tf.Variable;
    position: tf.Variable;
    direction: tf.Variable;
    _target: tf.Tensor2D;
    indices: tf.Variable;
    active: tf.Variable;

    constructor(gameConfig: GameLayerConfig, gameOptimizerConfig: NEATConfig) {
        super();
        this.step = 0;
        this.B = gameConfig.B;
        this.T = gameConfig.T;
        this.C = gameConfig.C;
        this.mutationRate = gameOptimizerConfig.mutationRate;
        this.fitness = tf.variable(tf.zeros([this.B]));
        this.position = tf.variable(tf.zeros([this.B, this.C]));
        this.direction = tf.variable(tf.eye(this.C).expandDims(0).tile([this.B, 1, 1]));
        this._target = tf.randomUniform([this.T, this.C]);
        this.indices = tf.variable(tf.randomUniformInt([this.B], -gameConfig.boundingBoxLength, gameConfig.boundingBoxLength));
        this.active = tf.variable(tf.ones([this.B]));
    }

    applyGradients(variableGradients: NEATVariableGradients): void {
        if (this.active.equal(0).all().arraySync() == 1) {
            this.initializePopulation(variableGradients.weights);
        }
    }

    initializePopulation(weights: tf.LayerVariable[]) {
        tf.tidy(() => {
            const bestIndex = this.fitnessScores.argMax();
            for (let i = 0; i < weights.length; i++) {
                const newWeights = weights[i].read().gather(bestIndex).tile([this.B, 1, 1]) as tf.Tensor3D;
                weights[i].write(this.mutate(newWeights));
            }
        })
    }

    mutate(weight: tf.Tensor3D): tf.Tensor3D {
        return weight.add(tf.randomNormal(weight.shape).mul(this.mutationRate));
    }

    getConfig(): NEATConfig {
        return {
            mutationRate: this.mutationRate
        };
    }
}

function main() {
    const optimizerConfig = {
        mutationRate: 0.8
    }
    const optimizer = new NEAT(settings.model, optimizerConfig);
    optimizer.applyGradients({
        weights: tf.tensor([1]),
        fitness: tf.tensor([1])
    })

}

main();