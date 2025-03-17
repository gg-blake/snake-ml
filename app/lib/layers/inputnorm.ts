import * as tf from '@tensorflow/tfjs';
import { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';
import '@tensorflow/tfjs-backend-webgl';
import GameLayer, { GameLayerConfig, IntermediateLayer } from './gamelayer';
import { dotProduct } from '../util';
import { settings } from "../../settings";


const projectBatchUOntoV: IntermediateLayer<[tf.Tensor3D, tf.Tensor3D], tf.Tensor4D> = (B, T, C, U, V) => tf.tidy(() => {
    const dot1 = tf.matMul(U, V, false, true);
    const dot2 = V.square().sum(-1).expandDims(-1).tile([1, 1, T]).transpose([0, 2, 1]) // (B, T, C)
    const div1 = tf.div(dot1, dot2).transpose([0, 2, 1]).expandDims(-1).tile([1, 1, 1, C]);
    const dot3 = V.expandDims(-2).tile([1, 1, T, 1]);
    const mul1 = tf.mul(div1, dot3).transpose([0, 2, 1, 3]);
    return mul1 as tf.Tensor4D
});

const calculateNearbyBody: IntermediateLayer<[tf.Tensor2D, tf.Tensor3D, tf.Tensor3D, number, number], tf.Tensor2D> = (B, T, C, position, direction, history, sensitivity, range) => tf.tidy(() => {
    // position: (B, C)
    // history: (B, T, C) <- T = global max food

    // Center body position at center and offset all other history position relative to the change
    const positionBroadcasted = position.expandDims(1).tile([1, T, 1]) as tf.Tensor3D; // (B, T, C)
    const historyRelativePosition = history.add<tf.Tensor3D>(positionBroadcasted.neg());

    // Calculate cosine similarity between each direction vector and every history position
    const dotProducts = tf.matMul(direction, historyRelativePosition, false, true);
    const directionMaginudes = direction.square().sum(-1).sqrt().expandDims(-2) as tf.Tensor3D; // (B, C, 1)
    const historyMagnitudes = historyRelativePosition.square().sum(-1).sqrt().expandDims(-1) as tf.Tensor2D; // (B, 1, T)
    const magnitudeDotProducts = tf.matMul(historyMagnitudes, directionMaginudes); // (B, C, T)
    const cosineSimilarityHistory = tf.div(dotProducts, magnitudeDotProducts.transpose([0, 2, 1])); // (B, C, T)
    
    // Check if any body parts are in the same direction of any of the direction vectors
    const positiveSimilarity = tf.any(cosineSimilarityHistory.greaterEqual(sensitivity), -1).cast('int32');
    const negativeSimilarity = tf.any(cosineSimilarityHistory.lessEqual(-sensitivity), -1).cast('int32');

    // Calculate the euclidean distance and mask it with cosine similarity (of a certain sensitivty)
    // Project each body part onto the given direction vector
    const bodyProjections = projectBatchUOntoV(B, T, C, historyRelativePosition, direction);
    const bodyProjectionsCapped = tf.where(bodyProjections.isNaN(), tf.fill(bodyProjections.shape, range), bodyProjections); // Replace NaN with Infinity to ignore for minimum calculation
    const relativeDistances = bodyProjectionsCapped.square().sum(-1).sqrt().slice([0, 1, 0], [B, T-1, C]).min(-2);
    
    const output = tf.div(relativeDistances.tile([1, 2]), tf.concat([positiveSimilarity, negativeSimilarity], -1)).clipByValue(-range, range); // (B, C * 2)
    return output as tf.Tensor2D
});

export const calculateNearbyTarget: IntermediateLayer<[tf.Tensor2D, tf.Tensor3D, tf.Tensor2D], tf.Tensor2D> = (B, T, C, position, direction, target) => tf.tidy(() => {
    // position: (B, C)
    // direction: (B, C, C)
    // target: (B, C)

    // Translate vector space, positioning position at the origin
    const targetRelativePosition = target.add<tf.Tensor2D>(position.neg()); // (B, C)
    const targetBroadcasted: tf.Tensor3D = targetRelativePosition.expandDims(1).tile([1, C, 1]); // (B, C, C)

    // Calculate cosine similarity between normalized targets and direction vectors
    const dot = tf.mul(direction, targetBroadcasted).sum(2); // (B, C)
    const dotNorm = tf.mul(tf.norm(direction, 'euclidean', 2), tf.norm(targetBroadcasted, 'euclidean', 2)); // (B, C)
    const similarity = tf.divNoNan(dot, dotNorm); // (B, C)
    
    return similarity as tf.Tensor2D
});

const raycast = (position: tf.Tensor2D, direction: tf.Tensor3D, scale: number): tf.Tensor2D => tf.tidy(() => {
    // position: (B, C)
    // direction: (B, C, C)
    const B = position.shape[0];
    const C = position.shape[1];

    const boxNormals = tf.concat([tf.eye(C), tf.eye(C).neg()], 0).expandDims(0).tile([B, 1, 1]) as tf.Tensor3D; // (B, 2C, C)
    const positionTile = position.expandDims(1).tile([1, 2 * C, 1]) as tf.Tensor3D; // (B, 2C, C)
    const dot1 = dotProduct<tf.Tensor3D>(positionTile, boxNormals, 2);
    const diff1 = tf.sub(scale, dot1) as tf.Tensor2D;
    const forwardDirectionTile = direction.slice([0, 0, 0], [B, 1, C]).tile([1, 2 * C, 1]) as tf.Tensor3D; // (B, 2C, C)
    const dot2 = dotProduct<tf.Tensor3D>(forwardDirectionTile, boxNormals, 2);
    const div1 = tf.div(diff1, dot2) as tf.Tensor2D;
    
    return div1
})

const closestDistance = (A: tf.Tensor2D): tf.Tensor1D => tf.tidy(() => {
    const mask = tf.where(A.greaterEqual(0), A, tf.fill(A.shape, Infinity));
    const min = tf.min(mask, 1) as tf.Tensor1D;
    
    return min;
})

const inBounds = (A: tf.Tensor2D): tf.Tensor2D => tf.tidy(() => {
    const B = A.shape[0];
    const C = Math.floor(A.shape[1] / 2);
    const firstHalf = A.slice([0, 0], [B, C]);
    const secondHalf = A.slice([0, C], [B, C]);
    const xor = tf.logicalXor<tf.Tensor2D>(firstHalf.less(0), secondHalf.less(0)).tile<tf.Tensor2D>([1, 2]);
    return xor
})

export const calculateNearbyBounds = (position: tf.Tensor2D, direction: tf.Tensor3D, scale: number) => tf.tidy(() => {
    const raycasted = raycast(position, direction, scale);
    const mask = inBounds(raycasted).cast<tf.Tensor2D>('float32');
    const distance = closestDistance(raycasted.mul(mask));
    
    return distance
})

type InputNormInputs = [
    tf.Tensor2D, // Position (B, C) [B, 0, C]
    tf.Tensor3D, // Direction (B, C, C) [B, 1-C+1, C]
    tf.Tensor2D, // Target (B, C) [B, C+1, C]
    tf.Tensor3D // History (B, T, C) [B, C+2-2C+2, C]
];

type InputNormSymbolicInputs = [
    tf.SymbolicTensor, // Position (B, C)
    tf.SymbolicTensor, // Direction (B, C, C)
    tf.SymbolicTensor, // Target (B, C)
    tf.SymbolicTensor // History (B, T, C)
];

export default class InputNorm extends GameLayer {
    constructor(config: LayerArgs, gameConfig: GameLayerConfig) {
        super(config, gameConfig);
        this.inputSpec = [
            {shape: [this.B, this.C]},
            {shape: [this.B, this.C, this.C]},
            {shape: [this.B, this.C]},
            {shape: [this.B, this.T, this.C]}
        ]
    }

    call(inputs: InputNormInputs): tf.Tensor3D {
        
        // (B, 2C+2, C)
        return tf.tidy(() => {
            // Food data processing and vectorizing (n_dims)
            const nearbyFood = calculateNearbyTarget(this.B, this.T, this.C, inputs[0], inputs[1], inputs[2]); // (B, C)
            // Body data processing and vectorizing (5) (n_dims * 2 - 1)
            const nearbyBody = calculateNearbyBody(this.B, this.T, this.C, inputs[0], inputs[1], inputs[3], 0.95, 1000).slice([0, 0], [this.B, 1]); // (B, 1)
            // Bounds data processing and vectorizing (3) (n_dims * 2 - 1)
            const nearbyBounds = calculateNearbyBounds(inputs[0], inputs[1], this.boundingBoxLength).expandDims(-1); // (B, 1)
            //nearbyBody.print()
            // Concatenate all sensory data to a single input vector
            return tf.concat([nearbyFood, nearbyBody, nearbyBounds], 1).expandDims(-1) as tf.Tensor3D; // (B, C + 2)
        })
    }

    computeOutputShape(inputShape: tf.Shape[]): tf.Shape | tf.Shape[] {
        return [this.B, this.C+2, 1];
    }
    
    static get className() {
        return 'InputNorm';
    }
}



tf.serialization.registerClass(InputNorm);