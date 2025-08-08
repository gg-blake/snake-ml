import * as tf from "@tensorflow/tfjs";
import { LayerArgs } from "@tensorflow/tfjs-layers/dist/engine/topology";
import "@tensorflow/tfjs-backend-webgl";
import {
    Fitness,
    Config,
    LogicInputs,
    LogicOutputs,
    LayerCallbackConfig,
} from "./types";
import { targetPlanarAngleDecomposition } from "./augment";
import { dotProduct, generatePlaneIndices } from "../../util";

interface WeightedFitnessParams {
    a: number;
    b: number;
    c: number;
    min: number;
    max: number;
}

function fitness(
    config: Config,
): (X: Fitness<tf.Tensor>) => Fitness<tf.Tensor> {
    const { a, b, c, min, max } = config.fitnessGraphParams;
    return function (X: Fitness<tf.Tensor>): Fitness<tf.Tensor> {
        return X.mul(a)
            .div(X.add(1).square().mul(b).add(c))
            .clipByValue(min, max) as Fitness<tf.Tensor>;
    };
}

const logic: LayerCallbackConfig<
    tf.Tensor,
    LogicInputs<tf.Tensor>,
    LogicOutputs<tf.Tensor>
> = (inputs, config): LogicOutputs<tf.Tensor> => {
    const [B, T, C] = config.batchInputShape! as number[];

    const { ttl, boundingBoxLength } = config;
    return tf.tidy(() => {
        //const outOfBounds = sensoryData.slice([0, this.inputLayerSize - 1], [this.B, 1]).squeeze().notEqual(0).cast('int32');
        //sensoryData.slice([0, this.inputLayerSize - 1], [this.B, 1]).print()
        //sensoryData.print()
        const outOfBounds = inputs[0]
            .abs()
            .greater(boundingBoxLength)
            .any(1)
            .logicalNot()
            .cast("int32");
        //outOfBounds.print();
        const gatheredTargets = inputs[2].gather(inputs[3]);
        const targetAngles = targetPlanarAngleDecomposition(
            config,
            inputs[0],
            inputs[1],
            gatheredTargets,
            generatePlaneIndices(C),
        )
            .div(Math.PI)
            .abs();
        const targetDirectionAverage = tf
            .sub(0.5, targetAngles.sum(-1).div(C - 1))
            .mul(2) as tf.Tensor1D; // (B,)
        const positionDifference = gatheredTargets.sub(
            inputs[0],
        ) as tf.Tensor2D;
        const distance = dotProduct<tf.Tensor2D>(
            positionDifference,
            positionDifference,
            -1,
        ).sqrt();
        const distanceWeighted = tf.div(1, distance.clipByValue(1, Infinity));

        const distanceInitial = tf
            .squaredDifference(inputs[7], gatheredTargets)
            .sum(-1)
            .sqrt();
        const distanceFinal = tf
            .squaredDifference(inputs[0], gatheredTargets)
            .sum(-1)
            .sqrt();
        const distanceDelta = distanceInitial
            .sub(distanceFinal)
            .clipByValue(-1, 1)
            .mul(inputs[5]) as tf.Tensor1D;
        const distanceDeltaAverage = distanceDelta
            .sum()
            .div(inputs[5].sum(-1)) as tf.Scalar;
        const distanceDeltaWeight = distanceDelta
            .sub(distanceDeltaAverage)
            .div(distanceDelta.max().sub(distanceDelta.min()))
            .mul(2)
            .sub(1) as tf.Tensor1D;

        const fitnessDelta = fitness(config)(distanceDelta).mul(
            inputs[5],
        ) as tf.Tensor1D;

        const touchingFood = distanceFinal.lessEqual(1).cast("float32");
        const nextFitness = inputs[4]
            .add(fitnessDelta)
            .add(
                touchingFood.mul(ttl * 2).pow(inputs[3].add(1)),
            ) as tf.Tensor1D;

        const isAliveMask = inputs[5].mul(outOfBounds) as tf.Tensor1D;

        const nextTargetIndices = inputs[3].add(
            touchingFood.cast("int32"),
        ) as tf.Tensor1D;
        return [nextFitness, nextTargetIndices, isAliveMask];
    });
};

export { logic };
