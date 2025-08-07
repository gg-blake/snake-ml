import * as tf from "@tensorflow/tfjs";
import { LayerArgs } from "@tensorflow/tfjs-layers/dist/engine/topology";
import "@tensorflow/tfjs-backend-webgl";
import { dotProduct, generatePlaneIndices } from "../../util";
import {
    LayerCallbackConfig,
    TensorOrArray,
    MovementInputs,
    MovementOutputs,
    Direction,
    Position,
    Config
} from "./types";

const rotateBatch = <T extends tf.Tensor | tf.Variable>(
    config: Config,
    direction: Direction<T>,
    planeIndices: tf.Tensor2D,
    theta: tf.Tensor1D,
    epsilon: number = 1e-8,
): Direction<T> =>
    tf.tidy(() => {
        const [B, T, C] = config.batchInputShape! as number[];
        // direction: (B, C, C)
        // planeIndices: (B, 2)
        // theta: (B,)

        const _identity: T = tf.eye(C).expandDims().tile([B, 1, 1]);
        const _indicesAugmented1 = tf
            .range(0, B)
            .expandDims(-1)
            .tile([1, 2])
            .expandDims(-1)
            .concat(planeIndices.expandDims(-1), -1);
        const _rotationSquareIndices = planeIndices
            .concat(planeIndices.reverse(1), 1)
            .expandDims(-1);
        const _indicesAugmented2 = _indicesAugmented1
            .tile([1, 2, 1])
            .concat(_rotationSquareIndices, 2)
            .reshape([4 * B, 3])
            .cast("int32");
        const _trig = tf
            .stack([
                tf.cos(theta),
                tf.cos(theta),
                tf.sin(theta).neg(),
                tf.sin(theta),
            ])
            .transpose()
            .reshape([4 * B]);
        const _rotation: TensorOrArray<T, tf.Rank.R3> = tf.tensorScatterUpdate(
            _identity,
            _indicesAugmented2,
            _trig,
        ) as TensorOrArray<T, tf.Rank.R3>;

        const _result = tf.matMul<tf.Tensor3D>(
            _rotation,
            direction,
            false,
            true,
        );

        // Values close to integer with difference of epsilon stick to closest integer value

        return _result as Direction<T>;
    });

const movement: LayerCallbackConfig<
    tf.Tensor,
    MovementInputs<tf.Tensor>,
    MovementOutputs<tf.Tensor>
> = (inputs: MovementInputs<tf.Tensor>, config) => {
    const [B, T, C] = config.batchInputShape! as number[];

    const half = generatePlaneIndices(B);
    const planeIndices = half.reverse(-1).concat(half, 0); // (2 * (C - 1), 2)
    const indices = planeIndices.gather(inputs[2].argMax(1)); // (B, 2)
    const nextDirections: Direction<tf.Tensor> = rotateBatch(
        config,
        inputs[1],
        indices,
        tf.ones([B]).mul(-Math.PI / 8),
    );
    const nextVelocity = nextDirections.slice([0, 0], [B, 1]).squeeze([1]);
    
    const nextPositions: Position<tf.Tensor> = inputs[0].add(
        nextVelocity.mul(config.stepSize).mul(inputs[3].expandDims(-1).tile([1, C])),
    );
    return [nextPositions, nextDirections];
};

/*const movement = (config: LayerArgs & GL.Config): MovementFn => {
    const wrapperFn = (inputs: MovementInputs) => tf.tidy(() => _movement(inputs, config));
    return wrapperFn
}*/

export { movement };
