import * as tf from '@tensorflow/tfjs';
import { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';
import '@tensorflow/tfjs-backend-webgl';
import GameLayer, * as GL from './gamelayer';
import { targetPlanarAngleDecomposition } from './inputnorm';
import { dotProduct, generatePlaneIndices } from '../util';

interface WeightedFitnessParams {
    a: number;
    b: number;
    c: number;
    min: number;
    max: number;
}

function fitness(config: GL.Config): (X: GL.Fitness<tf.Tensor>) => GL.Fitness<tf.Tensor> {
    const { a, b, c, min, max } = config.fitnessGraphParams;
    return function(X: GL.Fitness<tf.Tensor>): GL.Fitness<tf.Tensor> {
        return X
            .mul(a)
            .div(
                X.add(1)
                    .square()
                    .mul(b)
                    .add(c)
            )
            .clipByValue(min, max) as GL.Fitness<tf.Tensor>;
    }
}
    
type LogicInputs = [
    GL.Position<tf.Tensor>, // Position
    GL.Direction<tf.Tensor>, // Direction
    GL.Position<tf.Tensor>, // Target Position
    GL.TargetIndices<tf.Tensor>, // Target Index
    GL.Fitness<tf.Tensor>, // Fitness
    GL.Active<tf.Tensor>, // Active
    tf.Tensor2D, // Input Norms Layer Output
    GL.Position<tf.Tensor> // Previous Position
];

export default class Logic extends GameLayer {
    planeIndices: tf.Tensor2D;

    constructor(config: LayerArgs & GL.Config) {
        super(config, Logic.className);
        const [ B, T, C ] = this.config.batchInputShape! as number[];
        this.planeIndices = generatePlaneIndices(C);
    }

    call(inputs: LogicInputs): [tf.Tensor1D, tf.Tensor1D, tf.Tensor1D] {
        const [ B, T, C ] = this.config.batchInputShape! as number[];
        const { ttl, boundingBoxLength } = this.config;
        return tf.tidy(() => {
            //const outOfBounds = sensoryData.slice([0, this.inputLayerSize - 1], [this.B, 1]).squeeze().notEqual(0).cast('int32');
            //sensoryData.slice([0, this.inputLayerSize - 1], [this.B, 1]).print()
            //sensoryData.print()
            const outOfBounds = inputs[0].abs().greater(boundingBoxLength).any(1).logicalNot().cast('int32');
            //outOfBounds.print();
            const targetAngles = targetPlanarAngleDecomposition(this.config, inputs[0], inputs[1], inputs[2], this.planeIndices).div(Math.PI).abs();
            const targetDirectionAverage = tf.sub(0.5, targetAngles.sum(-1).div(C - 1)).mul(2) as tf.Tensor1D; // (B,)
            const positionDifference = inputs[2].sub(inputs[0]) as tf.Tensor2D;
            const distance = dotProduct<tf.Tensor2D>(positionDifference, positionDifference, -1).sqrt();
            const distanceWeighted = tf.div(1, distance.clipByValue(1, Infinity));

            const distanceInitial = tf.squaredDifference(inputs[7], inputs[2]).sum(-1).sqrt();
            const distanceFinal = tf.squaredDifference(inputs[0], inputs[2]).sum(-1).sqrt();
            const distanceDelta = distanceInitial.sub(distanceFinal).clipByValue(-1, 1).mul(inputs[5]) as tf.Tensor1D;
            const distanceDeltaAverage = distanceDelta.sum().div(inputs[5].sum(-1)) as tf.Scalar;
            const distanceDeltaWeight = distanceDelta.sub(distanceDeltaAverage).div(distanceDelta.max().sub(distanceDelta.min())).mul(2).sub(1) as tf.Tensor1D;
            
            const fitnessDelta = this.fitness(distanceDelta).mul(inputs[5]) as tf.Tensor1D;
            fitnessDelta.print();
            
            const touchingFood = distanceFinal.lessEqual(1).cast("float32");
            const nextFitness = inputs[4].add(fitnessDelta).add(touchingFood.mul(ttl*2).pow(inputs[3].add(1))) as tf.Tensor1D;
            
            const isAliveMask = inputs[5].mul(outOfBounds) as tf.Tensor1D;
            
            const nextTargetIndices = inputs[3].add(touchingFood.cast('int32')) as tf.Tensor1D;
            return [nextFitness, nextTargetIndices, isAliveMask];
        });
        
    }

    get fitness(): (X: tf.Tensor1D) => tf.Tensor1D {
        return fitness(this.config);
    }

    computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
        const [ B, T, C ] = this.config.batchInputShape! as number[];
        return [[B], [B], [B]]
    }

    static get className() {
        return 'Logic';
    }
}

tf.serialization.registerClass(Logic);