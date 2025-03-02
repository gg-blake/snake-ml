import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import { TTL, B, T, C, STARTING_SCORE, BOUNDS_SIZE } from "./constants";


const PLANE_INDICES = getPlaneIndices(C);

function normalize2D(A: tf.Tensor2D, dim: number): tf.Tensor2D {
    const norm = tf.norm(A, 'euclidean', dim);
    const normalized = A.div<tf.Tensor2D>(norm.expandDims(-1).tile([1, A.shape[1]]));
    return normalized
}

function getSubsetsOfSizeK<T>(arr: T[], k: number = 4): T[][] {
    const result: T[][] = [];

    function backtrack(start: number, subset: T[]) {
        // If the subset has reached size k, add it to the result
        if (subset.length === k) {
            result.push([...subset]);
            return;
        }

        for (let i = start; i < arr.length; i++) {
            // Include arr[i] in the subset
            subset.push(arr[i]);

            // Recursively generate subsets including this element
            backtrack(i + 1, subset);

            // Backtrack to remove the last element and try next option
            subset.pop();
        }
    }

    backtrack(0, []);
    return result;
}

function getPlaneIndices(numberOfDimensions: number): tf.Tensor2D {
    const _tensorList: tf.Tensor1D[] = getSubsetsOfSizeK<number>(Array.from(Array(numberOfDimensions).keys()), 2)
        .map((v: number[]) => tf.tensor(v, ...[,], 'int32'));
    return tf.stack<tf.Tensor1D>(_tensorList) as tf.Tensor2D;
}

const rotateBatch = (inputDirections: tf.Tensor3D, planeIndices: tf.Tensor2D, theta: tf.Tensor1D): tf.Tensor3D => tf.tidy(() => {
    // inputDirections: (B, C, C)
    // planeIndices: (B, 2)
    // theta: (B,)
    const B = inputDirections.shape[0];
    const C = inputDirections.shape[1];

    const _identity: tf.Tensor3D = tf.eye(C).expandDims().tile([B, 1, 1]);
    const _indicesAugmented1 = tf.range(0, B).expandDims(-1).tile([1, 2]).expandDims(-1).concat(planeIndices.expandDims(-1), -1);
    const _rotationSquareIndices = planeIndices.concat(planeIndices.reverse(1), 1).expandDims(-1);
    const _indicesAugmented2 = _indicesAugmented1.tile([1, 2, 1]).concat(_rotationSquareIndices, 2).reshape([4 * B, 3]).cast("int32");
    const _trig = tf.stack([tf.cos(theta), tf.cos(theta), tf.sin(theta).neg(), tf.sin(theta)]).transpose().reshape([4 * B]);
    const _rotation: tf.Tensor3D = tf.tensorScatterUpdate(_identity, _indicesAugmented2, _trig);
    const _result = tf.matMul<tf.Tensor3D>(_rotation, inputDirections, false, true);
    return _result
});

const projectOntoPlaneBatch = (P0: tf.Tensor3D, L0: tf.Tensor3D, n: tf.Tensor3D) => tf.tidy(() => {
    const L = tf.div<tf.Tensor2D>(L0, L0.norm('euclidean', 2).expandDims(2));
    const diff1 = tf.sub<tf.Tensor1D>(P0, L0);
    const dot1 = tf.mul(diff1, n).sum(2); // Batched row-pairwise dot product
    const dot2 = tf.mul(L, n).sum(2);
    const div1 = tf.div<tf.Tensor1D>(dot1, dot2);
    const P = tf.add(L0, tf.mul(L, div1.expandDims(2)));
    return P
});

const projectBatchUOntoV = (U: tf.Tensor3D, V: tf.Tensor3D): tf.Tensor4D => tf.tidy(() => {
    const dot1 = tf.matMul(U, V, false, true);
    const dot2 = V.square().sum(-1).expandDims(-1).tile([1, 1, T]).transpose([0, 2, 1]) // (B, T, C)
    const div1 = tf.div(dot1, dot2).transpose([0, 2, 1]).expandDims(-1).tile([1, 1, 1, C]);
    const dot3 = V.expandDims(-2).tile([1, 1, T, 1]);
    const mul1 = tf.mul(div1, dot3).transpose([0, 2, 1, 3]);
    return mul1 as tf.Tensor4D
});

const distanceFromPlaneBatch = (planePoint: tf.Tensor3D, normalPoint: tf.Tensor3D, position: tf.Tensor2D, velocity: tf.Tensor3D): [tf.Tensor2D, tf.Tensor1D] => tf.tidy(() => {
    // Offset the plane to ensure accurate distance
    const offsetPoint = tf.sub<tf.Tensor3D>(planePoint, position.expandDims(-2).tile([1, planePoint.shape[1], 1])); // (B, 2 * C, C)
    // Find the point that the line from the velocity vector intersects the plane
    const intersectionPoint = projectOntoPlaneBatch(offsetPoint, velocity, normalPoint);
    
    // Calculate the distance between the point on the plane and the velocity vector
    
    const distance = tf.add<tf.Tensor3D>(intersectionPoint, position.expandDims(1).tile([1, C * 2, 1]).neg()).square().sum<tf.Tensor2D>(2);
    const relativeIntersectionPoint = tf.add<tf.Tensor3D>(intersectionPoint, position.expandDims(1).tile([1, C * 2, 1]).neg())
    const relativeIntersectionPointPositive = relativeIntersectionPoint.slice([0, 0, 0], [B, C, C])
    const relativeIntersectionPointNegative = relativeIntersectionPoint.slice([0, C, 0], [B, C, C])
    const dot = tf.mul(relativeIntersectionPointNegative, relativeIntersectionPointPositive).sum(2); // (B, C)
    const dotNorm = tf.mul(tf.norm(relativeIntersectionPointNegative, 'euclidean', 2), tf.norm(relativeIntersectionPointPositive, 'euclidean', 2)); // (B, C)
    const similarityOpposite = tf.divNoNan(dot, dotNorm).less(0).slice([0, 0], [B, 1]).squeeze([1]) as tf.Tensor1D; // (B, C)
    return [distance, similarityOpposite]
});

const calculateNearbyBounds = (position: tf.Tensor2D, direction: tf.Tensor3D, scale: number) => tf.tidy(() => {
    const identity = tf.eye(C); // (C, C)
    const extendedIdentity = tf.concat([identity, identity.neg()]).mul<tf.Tensor2D>(scale).expandDims().tile<tf.Tensor3D>([B, 1, 1]); // (B, 2 * C, C)
    const extendedVelocity = tf.concat([direction, direction.neg()], 1); // (B, 2 * C, C)
    return distanceFromPlaneBatch(extendedIdentity, extendedIdentity.neg(), position, extendedVelocity) // (B, 2 * C)
})

const calculateNearbyBody = (position: tf.Tensor2D, direction: tf.Tensor3D, history: tf.Tensor3D, sensitivity: number, range: number) => tf.tidy(() => {
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
    const bodyProjections = projectBatchUOntoV(historyRelativePosition, direction);
    const bodyProjectionsCapped = tf.where(bodyProjections.isNaN(), tf.fill(bodyProjections.shape, range), bodyProjections); // Replace NaN with Infinity to ignore for minimum calculation
    const relativeDistances = bodyProjectionsCapped.square().sum(-1).sqrt().slice([0, 1, 0], [B, T-1, C]).min(-2);
    
    const output = tf.div(relativeDistances.tile([1, 2]), tf.concat([positiveSimilarity, negativeSimilarity], -1)).clipByValue(0, range); // (B, C * 2)
    return output
});

const calculateNearbyTarget = (position: tf.Tensor2D, direction: tf.Tensor3D, target: tf.Tensor2D) => tf.tidy(() => {
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
    
    return similarity
});

const forwardBatch = (params: BatchParamsTensor, targetPositions: tf.Tensor2D): BatchParamsTensor => tf.tidy(() => {
    const indexedFood = targetPositions.gather(params.targetIndices);

    // Food data processing and vectorizing (n_dims)
    const nearbyFood = calculateNearbyTarget(params.position, params.direction, indexedFood); // (B, C)

    // Body data processing and vectorizing (5) (n_dims * 2 - 1)
    const nearbyBody = calculateNearbyBody(params.position, params.direction, params.history, 0.95, 1000); // (B, C * 2)
    
    // Bounds data processing and vectorizing (3) (n_dims * 2 - 1)
    const [nearbyBounds, outOfBounds] = calculateNearbyBounds(params.position, params.direction, BOUNDS_SIZE); // (B, C * 2)
    
    // Concatenate all sensory data to a single input vector
    const inputs = tf.concat([nearbyFood, nearbyBounds, nearbyBody], 1).expandDims(-1); // (B, C * 5)
    //const inputsArray = inputs.arraySync() as number[][];
    
    // Feed forward linear layers (no bias)
    const hidden = tf.matMul(params.weights[0], inputs, false, false)
    //const hiddenReLU = tf.clipByValue(hidden, 0, 1); // ReLU activation function
    const logits = tf.matMul(params.weights[1], hidden, false, false)
    //const logitsReLU = tf.clipByValue(logits, 0, 1); // ReLU activation function
    const logitsFlattened = logits.squeeze([2]) as tf.Tensor2D; // (B, (((C-1)*C)/2))
    
    const logitsNormalized = normalize2D(logitsFlattened, 1);
    
    const indices = PLANE_INDICES.gather(logitsNormalized.sub(0.5).abs().argMax(1));
    const angles = logitsNormalized.gather(logitsNormalized.sub(0.5).abs().argMax(1), 1, 1) as unknown as tf.Tensor1D;
    const nextDirections = rotateBatch(params.direction, indices, angles);
    
    
    const nextVelocity: tf.Tensor2D = nextDirections.slice([0, 0], [B, 1]).squeeze([1]);
    const nextPositions: tf.Tensor2D = params.position.add(nextVelocity);
    const nextDistances = tf.squaredDifference(nextPositions, indexedFood).sum(1).sqrt(); // Euclidean distance
    const previousDistances = tf.squaredDifference(params.position, indexedFood).sum(1).sqrt();
    const distanceDifferences = tf.sub(previousDistances, nextDistances);
    
    const touchingFood = nextDistances.lessEqual(3).cast("int32");
    const improvingMask = distanceDifferences.greater(1).mul(1).add(touchingFood.mul(TTL));
    const degradingMask = distanceDifferences.lessEqual(1).mul(-5);
    
    const isAliveMask = params.fitness.greater(params.targetIndices.mul(TTL)).cast('int32').mul(params.alive).mul(outOfBounds) as tf.Tensor1D;
    
    
    const nextFitnesses: tf.Tensor1D = params.fitness.add(improvingMask.add(degradingMask).mul(isAliveMask));
    
    const nextTargetIndices = params.targetIndices.add(touchingFood) as tf.Tensor1D;

    const cutoffMask = tf.range(0, T).expandDims().tile([B, 1]).less(nextTargetIndices.expandDims(-1).tile([1, T]).add(STARTING_SCORE)).cast('int32'); // (B, T)
    const cutoffKeep = tf.concat([nextPositions.expandDims(1), params.history], 1).slice([0, 0, 0], [B, T, C]); // (B, T, C)
    const nextHistory = cutoffKeep.div(cutoffMask.expandDims(1).tile([1, C, 1]).transpose([0, 2, 1])) as tf.Tensor3D; // (B, T, C)

    // Dispose of old param tensors
    params.position.dispose()
    params.direction.dispose()
    params.history.dispose();
    params.targetIndices.dispose();
    params.fitness.dispose();
    params.alive.dispose();
    
    return { 
        position: tf.keep(nextPositions), 
        direction: tf.keep(nextDirections), 
        fitness: tf.keep(nextFitnesses), 
        targetIndices: tf.keep(nextTargetIndices), 
        history: tf.keep(nextHistory), 
        weights: params.weights.map(w => tf.keep(w)),
        alive: tf.keep(isAliveMask)
    }
});

const generateWeights = (weightsA: tf.Tensor3D[], crossoverProbability: number, differentialWeight: number): BatchParamsTensor => tf.tidy(() => {
    const weightsB: tf.Tensor3D[] = [];
    for (let weightsNum = 0; weightsNum < weightsA.length; weightsNum++) {
        const T = weightsA[weightsNum].shape[1];
        const C = weightsA[weightsNum].shape[2];
        
        const ranges1 = tf.range(0, B, ...[,], "int32").expandDims().tile([B, 1]);
        const ranges2 = tf.range(0, B, ...[,], "int32").expandDims().tile([B, 1]).transpose();
        const upper = tf.greater(ranges1, ranges2).cast('int32').mul(B).reverse(0);
        const idx = tf.add(ranges1, ranges2).sub(upper).slice([0, 1], [B, B-1]);
        const indices = Array.from(Array(B-1).keys(), (v: number) => v+1);
        tf.util.shuffle(indices)
        const randomIndices = tf.tensor(indices, ...[,], "int32").slice([0], [3]).expandDims().tile([B, 1]);

        const idx2 = tf.gather(idx, randomIndices, 1, 1).cast("int32"); // (B, 3)
        const uniqueWeights = tf.gather(weightsA[weightsNum], idx2) as unknown as tf.Tensor4D; // (B, 3, T, C)
        const rValues = tf.randomUniform(weightsA[weightsNum].shape, ...[,,], "float32") // (B, T, C)
        const mask = rValues.less(crossoverProbability).cast('int32'); // (B, T, C)
        const mask2 = tf.oneHot(tf.randomUniformInt([B, T], 0, C), C).cast('int32'); // (B, T, C)
        const positiveMask = mask.add(mask2).clipByValue(0, 1);

        const negativeMask = positiveMask.notEqual(1);
        
        const p0 = tf.slice(uniqueWeights, [0, 0, 0, 0], [B, 1, T, C]).squeeze<tf.Tensor3D>([1]);
        const p1 = tf.slice(uniqueWeights, [0, 1, 0, 0], [B, 1, T, C]).squeeze<tf.Tensor3D>([1]);
        const p2 = tf.slice(uniqueWeights, [0, 2, 0, 0], [B, 1, T, C]).squeeze<tf.Tensor3D>([1]);

        const updateWeightRule = p1.sub<tf.Tensor3D>(p2).mul<tf.Tensor3D>(differentialWeight).add<tf.Tensor3D>(p0);
        const dontUpdateWeightRule = negativeMask.mul<tf.Tensor3D>(weightsA[weightsNum]);
        const transformedWeights = tf.keep(updateWeightRule.mul<tf.Tensor3D>(positiveMask).add<tf.Tensor3D>(dontUpdateWeightRule));

        weightsB.push(transformedWeights);
    }

    return {
        fitness: tf.keep(tf.ones([B]).mul(TTL) as tf.Tensor1D),
        position: tf.keep(tf.zeros([B, C]) as tf.Tensor2D),
        targetIndices: tf.keep(tf.zeros([B], 'int32') as tf.Tensor1D),
        direction: tf.keep(tf.eye(C).expandDims(0).tile([B, 1, 1]) as tf.Tensor3D),
        history: tf.keep(tf.zeros([B, T, C - 1]).concat(tf.range(0, T).reshape([1, T]).tile([B, 1]).expandDims(-1).neg(), -1) as tf.Tensor3D),
        weights: weightsB,
        alive: tf.keep(tf.ones([B], 'int32') as tf.Tensor1D)
    }
});



const compareWeights = (fitnessesA: tf.Tensor1D, weightsA: tf.Tensor3D[], fitnessesB: tf.Tensor1D, weightsB: tf.Tensor3D[]): BatchParamsTensor => tf.tidy(() => {
    const fitnessesStacked: tf.Tensor2D = tf.stack<tf.Tensor1D>([fitnessesA, fitnessesB], 1) as tf.Tensor2D;
    const argMaxFitnesses = tf.argMax(fitnessesStacked, 1);

    const weightsC = [];
    for (let weightsNum = 0; weightsNum < weightsA.length; weightsNum++) {
        const weightsStacked: tf.Tensor4D = tf.stack<tf.Tensor3D>([weightsA[weightsNum], weightsB[weightsNum]], 1) as tf.Tensor4D;
        const fittestWeightsPartial: tf.Tensor3D = tf.keep(tf.gather(weightsStacked, argMaxFitnesses, 1, 1) as unknown as tf.Tensor3D);
        weightsC.push(fittestWeightsPartial);
    }

    return {
        fitness: tf.keep(tf.ones([B]).mul(TTL) as tf.Tensor1D),
        position: tf.keep(tf.zeros([B, C]) as tf.Tensor2D),
        targetIndices: tf.keep(tf.zeros([B], 'int32') as tf.Tensor1D),
        direction: tf.keep(tf.eye(C).expandDims(0).tile([B, 1, 1]) as tf.Tensor3D),
        history: tf.keep(tf.zeros([B, T, C - 1]).concat(tf.range(0, T).reshape([1, T]).tile([B, 1]).expandDims(-1).neg(), -1) as tf.Tensor3D),
        weights: weightsC,
        alive: tf.keep(tf.ones([B], 'int32') as tf.Tensor1D)
    }
});

interface BatchParamsTensor {
    position: tf.Tensor2D;
    direction: tf.Tensor3D;
    fitness: tf.Tensor1D;
    targetIndices: tf.Tensor1D;
    history: tf.Tensor3D;
    weights: tf.Tensor3D[];
    alive: tf.Tensor1D;
}

interface BatchParamsArray {
    position: number[][];
    direction: number[][][];
    fitness: number[];
    targetIndices: number[];
    history: number[][][];
    weights: number[][][][];
    alive: number[];
}

const batchArraySync = (a: BatchParamsTensor): BatchParamsArray => tf.tidy(() => {
    return {
        position: a.position.arraySync() as number[][],
        direction: a.direction.arraySync() as number[][][],
        fitness: a.fitness.arraySync() as number[],
        targetIndices: a.targetIndices.arraySync() as number[],
        history: a.history.arraySync() as number[][][],
        weights: a.weights.map(w => w.arraySync() as number[][][]),
        alive: a.alive.arraySync() as number[]
    }
});

export { forwardBatch, batchArraySync, compareWeights, generateWeights }
export type { BatchParamsTensor, BatchParamsArray }