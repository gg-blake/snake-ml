import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';

const NUMBER_OF_DIMENSIONS = 3
const PLANE_INDICES = getPlaneIndices(NUMBER_OF_DIMENSIONS);

function normalize<T extends tf.Tensor<tf.Rank>>(A: tf.Tensor, dim: number): T {
    const norm = tf.norm(A, 'euclidean', dim);
    const normalized = A.div<T>(norm);
    return normalized
}

function getSubsetsOfSizeK<T>(arr: T[], k: number = 4): T[][] {
    let result: T[][] = [];

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

function getPlaneIndices(numberOfDimensions: number) {
    const _tensorList: tf.Tensor[] = getSubsetsOfSizeK<number>(Array.from(Array(numberOfDimensions).keys()), 2)
        .map((v: number[]) => tf.tensor(v, ...[,], 'int32'));
    return tf.stack(_tensorList);
}


function rotateBatch(inputDirections: tf.Tensor3D, planeIndices: tf.Tensor3D, angles: tf.Tensor2D): tf.Tensor3D {
    let _normalVectors = inputDirections;
    const _basisVectors = _normalVectors.gather(planeIndices);
    const _basisVectorsTransposed = _basisVectors.transpose([0, _basisVectors.rank - 1, _basisVectors.rank - 2]);

    const _b = inputDirections.shape[0]
    const _m = _basisVectorsTransposed.shape[_basisVectors.rank - 1];
    const _n = _basisVectorsTransposed.shape[_basisVectors.rank - 2];

    const _vProjected = tf.matMul(_basisVectors, _normalVectors);
    
    const _batchCosine: tf.Tensor2D = tf.cos<tf.Tensor2D>(angles);
    const _batchSine: tf.Tensor2D = tf.sin<tf.Tensor2D>(angles);
    
    
    const _rotationMatrixIdentity: tf.Tensor4D = tf.eye(_m).expandDims(0).expandDims(0).tile([_b, angles.shape[0], 1, 1]);
    const _range: tf.Tensor3D = tf.range(0, _rotationMatrixIdentity.shape[1], ...[,], "int32").expandDims().expandDims().tile([1, 4, 1]).transpose([0, 2, 1]).reshape([_b, 1, _rotationMatrixIdentity.shape[0] * 4]).transpose([0, 2, 1])
    const _indices: tf.Tensor3D = tf.tensor2d([[0, 0], [0, 1], [1, 0], [1, 1]], ...[,], "int32").expandDims().tile([_m, _rotationMatrixIdentity.shape[1], 1]);
    const _repeatedIndices = tf.concat([_range, _indices], 2);
    
    
    const _rot: tf.Tensor2D = tf.stack<tf.Tensor2D>([_batchCosine, _batchSine.neg(), _batchSine, _batchCosine]).transpose([0, 2, 1]).reshape([_m, 1, _rotationMatrixIdentity.shape[0] * 4]).squeeze([1]);
    

    const _rotationMatrix = tf.tensorScatterUpdate(_rotationMatrixIdentity, _repeatedIndices, _rot);
    
    const _vRotatedProjected = tf.matMul(_rotationMatrix, _vProjected);
    const _vRotated = tf.matMul(_basisVectors.transpose([0, 2, 1]), _vRotatedProjected);

    return normalize<tf.Tensor3D>(_vRotated.slice([0, 0], [_b, NUMBER_OF_DIMENSIONS]).sum(1), 2);
}

function projectOntoPlaneBatch(P0: tf.Tensor3D, L0: tf.Tensor3D, n: tf.Tensor3D) {
    const L = tf.div<tf.Tensor2D>(L0, L0.norm('euclidean', 2).expandDims(2));
    const diff1 = tf.sub<tf.Tensor1D>(P0, L0);
    const dot1 = tf.mul(diff1, n).sum(2); // Batched row-pairwise dot product
    const dot2 = tf.mul(L, n).sum(2);
    const div1 = tf.div<tf.Tensor1D>(dot1, dot2);
    const P = tf.add(L0, tf.mul(L, div1.expandDims(2)));
    return P
}

function distanceFromPlaneBatch(planePoint: tf.Tensor3D, normalPoint: tf.Tensor3D, position: tf.Tensor2D, velocity: tf.Tensor3D): tf.Tensor2D {
    // Offset the plane to ensure accurate distance
    const offsetPoint = tf.sub<tf.Tensor3D>(planePoint, position);
    // Find the point that the line from the velocity vector intersects the plane
    const intersectionPoint = projectOntoPlaneBatch(offsetPoint, velocity, normalPoint);
    // Calculate the distance between the point on the plane and the velocity vector
    const distance = tf.sub<tf.Tensor2D>(intersectionPoint, velocity).square().sum<tf.Tensor2D>(2).sqrt();
    return distance
}

function cosineSimilarity(A: tf.Tensor, B: tf.Tensor, dim: number): tf.Tensor {
    const euclideanNormA = tf.square(A).sum(dim).sqrt();
    const euclideanNormB = tf.square(B).sum(dim).sqrt();
    const productTop = tf.mul(A, B).sum(dim);
    const productBottom = tf.mul(euclideanNormA, euclideanNormB);
    const diff = tf.divNoNan(productTop, productBottom);
    return diff
}

function forwardBatch(positions: tf.Tensor2D, targetIndices: tf.Tensor1D, targetPositions: tf.Tensor2D, directions: tf.Tensor3D, fitnesses: tf.Tensor1D) {
    const _b = positions.shape[0]
    const indexedFood = targetPositions.gather(targetIndices);

    // Extend basis vectors to negative axis
    const signedVelocity = tf.concat([directions, directions.mul(-1)]);

    // Extend unit vectors to negative axis
    const signedUnitVectors = tf.concat([tf.eye(NUMBER_OF_DIMENSIONS), tf.eye(NUMBER_OF_DIMENSIONS).mul(-1)]);

    // Food data processing and vectorizing (n_dims * 2)
    const foodNormals = cosineSimilarity(positions.sub(indexedFood).expandDims(0).tile([NUMBER_OF_DIMENSIONS * 2, 1]), signedVelocity, 1);
    const nearbyFood = foodNormals.clipByValue(-1, 1);
    const foodDistance = tf.squaredDifference(positions, indexedFood).sum(-1).sqrt();

    // Body data processing and vectorizing (5) (n_dims * 2 - 1)
    /*const bodyNormals = signedVelocity.add(positions.expandDims(0).tile([this.numberOfDimensions * 2, 1]));
    const snakeHistoryHeadOmitted = snakeHistory.slice([0, 0], [snakeHistory.shape[0] - 1, this.numberOfDimensions]); // Omit the head of the snake
    const bodyDistances = cdist(bodyNormals, snakeHistoryHeadOmitted);
    const nearbyBody = tf.any(tf.concat([bodyDistances.slice([0], [this.numberOfDimensions]), bodyDistances.slice([this.numberOfDimensions + 1], [this.numberOfDimensions - 1])]).equal(0), 1).cast('float32');*/

    // Bounds data processing and vectorizing (3) (n_dims * 2 - 1)
    // TODO: Add new raycasting function to determine bounds
    const identity = tf.eye(NUMBER_OF_DIMENSIONS);
    const extendedIdentity = tf.concat([identity, identity.neg()], 1).mul<tf.Tensor3D>(10);
    const extendedVelocity = tf.concat([directions, directions.neg()], 1);
    const nearbyBounds = distanceFromPlaneBatch(extendedIdentity, extendedIdentity.neg(), positions, extendedVelocity).softmax(1)

    const inputs = tf.concat([nearbyFood, nearbyBounds], 1);

    const feedForward = new tf.Sequential({
        layers: [
            tf.layers.dense({
                units: 100,
                batchInputShape: inputs.shape,
                useBias: false,  // Matches PyTorch's bias=False
                activation: 'relu' // ReLU activation
            }),
            tf.layers.dense({
                units: Math.floor((NUMBER_OF_DIMENSIONS - 1) * NUMBER_OF_DIMENSIONS / 2),
                useBias: false,  // Matches PyTorch's bias=False
                activation: 'relu' // ReLU activation
            })
        ]
    })

    const logits: tf.Tensor2D = (feedForward.predict(inputs) as tf.Tensor3D).squeeze([1]);
    const angles = normalize<tf.Tensor2D>(logits, 1).sub(0.5).mul<tf.Tensor2D>(Math.PI);
    const nextDirections = rotateBatch(directions, PLANE_INDICES.expandDims().tile([_b]), angles);
    const nextVelocity: tf.Tensor2D = nextDirections.slice([0, 0], [_b, 1]).squeeze([1]);
    const nextPositions: tf.Tensor2D = positions.add(nextVelocity);

    
    const nextDistances = tf.squaredDifference(nextPositions, targetPositions).sum(1).sqrt(); // Euclidean distance
    const previousDistances = tf.squaredDifference(positions, targetPositions).sum(1).sqrt();
    const distanceDifferences = tf.sub(previousDistances, nextDistances);
    const improvingMask = distanceDifferences.greater(0).mul(1);
    const degradingMask = distanceDifferences.lessEqual(0).mul(-5);
    const nextFitnesses: tf.Tensor1D = fitnesses.add(improvingMask.add(degradingMask));

    return { nextPositions, nextDirections, nextFitnesses }
}