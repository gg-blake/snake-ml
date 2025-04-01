import * as tf from '@tensorflow/tfjs';
import { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';
import '@tensorflow/tfjs-backend-webgl';
import GameLayer, { GameLayerConfig, IntermediateLayer } from './gamelayer';
import { targetPlanarAngleDecomposition } from './inputnorm';
import { dotProduct, generatePlaneIndices } from '../util';

const calculateFitness: IntermediateLayer<[tf.Tensor1D, number, number, number, number, number], tf.Tensor1D> = (B, T, C, X, a, b, c, min, max): tf.Tensor1D => X
    .mul(a)
    .div(
        X.add(1)
            .square()
            .mul(b)
            .add(c)
    )
    .clipByValue(min, max) as tf.Tensor1D;

type LogicInputs = [
    tf.Tensor2D, // Position
    tf.Tensor3D, // Direction
    tf.Tensor2D, // Target Position
    tf.Tensor1D, // Target Index
    tf.Tensor1D, // Fitness
    tf.Tensor1D, // Active
    tf.Tensor2D // Input Norms Layer Output
];

interface LogicState {
    fitnessDelta: tf.Variable<tf.Rank.R1>;
}

export default class Logic extends GameLayer {
    maxFitness: tf.Variable<tf.Rank.R1>;
    planeIndices: tf.Tensor2D;
    state: LogicState;

    constructor(config: LayerArgs, gameConfig: GameLayerConfig) {
        super(config, gameConfig);
        this.maxFitness = tf.variable(tf.ones([gameConfig.B]).mul(gameConfig.TTL));
        this.planeIndices = generatePlaneIndices(this.C);
        this.state = {
            fitnessDelta: tf.variable(tf.zeros([this.B], "float32"))
        }
    }

    call(inputs: LogicInputs): [tf.Tensor1D, tf.Tensor1D, tf.Tensor1D] {
        const P = this.fitnessGraphParams;
        return tf.tidy(() => {
            //const outOfBounds = sensoryData.slice([0, this.inputLayerSize - 1], [this.B, 1]).squeeze().notEqual(0).cast('int32');
            //sensoryData.slice([0, this.inputLayerSize - 1], [this.B, 1]).print()
            //sensoryData.print()
            const outOfBounds = inputs[0].abs().greater(this.boundingBoxLength).any(1).logicalNot().cast('int32');
            //outOfBounds.print();
            const targetAngles = targetPlanarAngleDecomposition(this.B, this.T, this.C, inputs[0], inputs[1], inputs[2], this.planeIndices).div(Math.PI).abs();
            const targetDirectionAverage = tf.sub(0.5, targetAngles.sum(-1).div(this.C - 1)).mul(2) as tf.Tensor1D; // (B,)
            const positionDifference = inputs[2].sub(inputs[0]) as tf.Tensor2D;
            const distance = dotProduct<tf.Tensor2D>(positionDifference, positionDifference, -1).sqrt();
            const distanceWeighted = tf.div(1, distance.clipByValue(1, Infinity));
            
            const fitnessDelta = calculateFitness(this.B, this.T, this.C, targetDirectionAverage, P.a, P.b, P.c, P.min, P.max).mul(inputs[5]).mul(5) as tf.Tensor1D;
            const nextDistance = tf.squaredDifference(inputs[0], inputs[2]).sum(1).sqrt();
            const touchingFood = nextDistance.lessEqual(1).cast("float32");
            const nextFitness = inputs[4].add(fitnessDelta).add(touchingFood.mul(this.TTL*2)) as tf.Tensor1D;
            
            const isAliveMask = inputs[5].mul(outOfBounds) as tf.Tensor1D;
            const maxFitness = tf.stack([this.maxFitness, nextFitness], 1).max(1) as tf.Tensor1D;
            this.state.fitnessDelta.assign(fitnessDelta);
            this.maxFitness.assign(maxFitness);
            const nextTargetIndices = inputs[3].add(touchingFood.cast('int32')) as tf.Tensor1D;
            return [nextFitness, nextTargetIndices, isAliveMask];
        });
        
    }

    computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
        return [[this.B], [this.B], [this.B]]
    }

    static get className() {
        return 'Logic';
    }
}

tf.serialization.registerClass(Logic);