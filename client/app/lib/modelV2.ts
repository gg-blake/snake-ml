import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';


const PLANE_INDICES = getPlaneIndices(4);

function normalize2D(A: tf.Tensor2D, dim: number): tf.Tensor2D {
    const norm = tf.norm(A, 'euclidean', dim);
    const normalized = A.div<tf.Tensor2D>(norm.expandDims(-1).tile([1, A.shape[1]]));
    return normalized
}

function normalize3D(A: tf.Tensor3D, dim: number): tf.Tensor3D {
    const norm = tf.norm(A, 'euclidean', 2);
    const normalized = A.div<tf.Tensor3D>(norm.expandDims(-1).tile([1, 1, A.shape[2]]));
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

function getPlaneIndices(numberOfDimensions: number): tf.Tensor2D {
    const _tensorList: tf.Tensor1D[] = getSubsetsOfSizeK<number>(Array.from(Array(numberOfDimensions).keys()), 2)
        .map((v: number[]) => tf.tensor(v, ...[,], 'int32'));
    return tf.stack<tf.Tensor1D>(_tensorList) as tf.Tensor2D;
}


function rotateBatch(inputDirections: tf.Tensor3D, planeIndices: tf.Tensor3D, angles: tf.Tensor2D): tf.Tensor3D {
    // inputDirections: (B, C, C)
    // planeIndices: (B, (((C-1)*C)/2), 2)
    // angles: (B, (((C-1)*C)/2))

    const B = angles.shape[0] // Population size
    const T = planeIndices.shape[2]; // =2
    const C = inputDirections.shape[1];
    
    let _normalVectors = inputDirections; // (B, C, C)
    const _basisVectors: tf.Tensor4D = tf.gather<tf.Tensor3D>(_normalVectors, planeIndices, 1, 1) as unknown as tf.Tensor4D; // (B, (((C-1)*C)/2), T, C)
    const _basisVectorsTransposed = _basisVectors.transpose([0, 1, _basisVectors.rank - 1, _basisVectors.rank - 2]); // (B, (((C-1)*C)/2), C, T)
    const _vProjected: tf.Tensor4D = tf.matMul(_basisVectors, _normalVectors.expandDims(1).tile([1, _basisVectors.shape[1], 1, 1])); // (B, (((C-1)*C)/2), T, C)
    const _batchCosine: tf.Tensor2D = tf.cos<tf.Tensor2D>(angles); // (B, (((C-1)*C)/2))
    const _batchSine: tf.Tensor2D = tf.sin<tf.Tensor2D>(angles); // (B, (((C-1)*C)/2))
    const _rotationMatrixIdentity: tf.Tensor4D = tf.eye(T).expandDims(0).tile([angles.shape[1], 1, 1]).expandDims(0).tile([B, 1, 1, 1]); // (B, (((C-1)*C)/2), T, T)
    const _range: tf.Tensor3D = tf.range(0, _rotationMatrixIdentity.shape[1], ...[,], "int32").expandDims(0).tile([4, 1]).transpose().reshape([1, (((C-1)*C)/2) * 4]).transpose().expandDims(0).tile([B, 1, 1]) // (B, (((C-1)*C)/2)*4, 1)
    const _rangeBatch = tf.range(0, B, ...[,],"int32").expandDims(-1).tile([1, (((C-1)*C)/2)*4]).expandDims(-1)
    const _indices: tf.Tensor3D = tf.tensor2d([[0, 0], [0, 1], [1, 0], [1, 1]], ...[,], "int32").expandDims().tile([B, _rotationMatrixIdentity.shape[1], 1]);
    const _repeatedIndices = tf.concat([_range, _indices], 2);
    const _repeatedIndicesBatch = tf.concat([_rangeBatch, _repeatedIndices], 2).reshape([B*(((C-1)*C)/2)*4, 4]);
    const _rot: tf.Tensor2D = tf.stack<tf.Tensor2D>([_batchCosine, _batchSine.neg(), _batchSine, _batchCosine], 1).transpose([0, 2, 1]).reshape([B, 1, _rotationMatrixIdentity.shape[1] * 4]).squeeze([1]).reshape([B*(((C-1)*C)/2)*4]);
    const _rotationMatrix = tf.tensorScatterUpdate(_rotationMatrixIdentity, _repeatedIndicesBatch, _rot);
    const _vRotatedProjected = tf.matMul(_rotationMatrix, _vProjected);
    const _vRotated = tf.matMul(_basisVectors.transpose([0, 1, 3, 2]), _vRotatedProjected);
    const _vRotatedNormalized = normalize3D(_vRotated.slice([0, 0, 0], [B, C]).sum(1), 2);
    
    return _vRotatedNormalized // (B, C, C)
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
    const offsetPoint = tf.sub<tf.Tensor3D>(planePoint, position.expandDims(-2).tile([1, planePoint.shape[1], 1])); // (B, 2 * C, C)
    // Find the point that the line from the velocity vector intersects the plane
    const intersectionPoint = projectOntoPlaneBatch(offsetPoint, velocity, normalPoint);
    // Calculate the distance between the point on the plane and the velocity vector
    const distance = tf.squaredDifference<tf.Tensor3D>(intersectionPoint, velocity).sum<tf.Tensor2D>(2).sqrt();
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
    const _b = positions.shape[0];
    const _c = positions.shape[1];
    const indexedFood = targetPositions.gather(targetIndices);

    // Extend basis vectors to negative axis
    const signedVelocity: tf.Tensor3D = tf.concat<tf.Tensor3D>([directions, directions.mul<tf.Tensor3D>(-1)], 1); // (B, 2 * C, C)

    // Extend unit vectors to negative axis
    const signedUnitVectors: tf.Tensor3D = tf.concat([tf.eye(_c), tf.eye(_c).mul(-1)]).expandDims().tile([_b, 1, 1]); // (B, 2 * C, C)

    // Food data processing and vectorizing (n_dims * 2)
    const foodNormals = cosineSimilarity(positions.sub(indexedFood).expandDims(1).tile([1, _c * 2, 1]), signedVelocity, 2); // (B, 2 * C)
    const nearbyFood = foodNormals.clipByValue(-1, 1);
    const foodDistance = tf.squaredDifference(positions, indexedFood).sum(-1).sqrt(); // (B,)

    // Body data processing and vectorizing (5) (n_dims * 2 - 1)
    /*const bodyNormals = signedVelocity.add(positions.expandDims(0).tile([this.numberOfDimensions * 2, 1]));
    const snakeHistoryHeadOmitted = snakeHistory.slice([0, 0], [snakeHistory.shape[0] - 1, this.numberOfDimensions]); // Omit the head of the snake
    const bodyDistances = cdist(bodyNormals, snakeHistoryHeadOmitted);
    const nearbyBody = tf.any(tf.concat([bodyDistances.slice([0], [this.numberOfDimensions]), bodyDistances.slice([this.numberOfDimensions + 1], [this.numberOfDimensions - 1])]).equal(0), 1).cast('float32');*/

    // Bounds data processing and vectorizing (3) (n_dims * 2 - 1)
    // TODO: Add new raycasting function to determine bounds
    const identity = tf.eye(_c);
    const extendedIdentity = tf.concat([identity, identity.neg()]).mul<tf.Tensor2D>(10).expandDims().tile<tf.Tensor3D>([_b, 1, 1]); // (B, 2 * C, C)
    const extendedVelocity = tf.concat([directions, directions.neg()], 1); // (B, 2 * C, C)
    const nearbyBounds: tf.Tensor2D = distanceFromPlaneBatch(extendedIdentity, extendedIdentity.neg(), positions, extendedVelocity).softmax<tf.Tensor2D>(1) // (B, 2 * C)

    const inputs = tf.concat([nearbyFood, nearbyBounds], 1); // (B, 4 * C)

    const feedForward = new tf.Sequential({
        layers: [
            tf.layers.dense({
                units: 100,
                batchInputShape: inputs.shape,
                useBias: false,  // Matches PyTorch's bias=False
                activation: 'relu' // ReLU activation
            }),
            tf.layers.dense({
                units: Math.floor((_c - 1) * _c / 2),
                useBias: false,  // Matches PyTorch's bias=False
                activation: 'relu' // ReLU activation
            })
        ]
    })

    const logits: tf.Tensor2D = feedForward.predict(inputs) as tf.Tensor2D; // (B, (((C-1)*C)/2))
    const angles = normalize2D(logits, 1).sub(0.5).mul<tf.Tensor2D>(Math.PI); // (B, (((C-1)*C)/2))
    const nextDirections = rotateBatch(directions, PLANE_INDICES.expandDims(0).tile([_b, 1, 1]), angles);
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

function main() {
    const B = 100; 
    const C = 4; // Number of dimensions
    const G = 4; // Number of food
    const positions: tf.Tensor2D = tf.zeros([B, C]);
    const targetIndices: tf.Tensor1D = tf.randomUniformInt([B], 0, G);
    const targets: tf.Tensor2D = tf.randomUniform([G, C], -10, 10);
    const directions: tf.Tensor3D = tf.randomUniform([B, C, C]);
    const fitnesses: tf.Tensor1D = tf.ones([B], "float32").mul(100); // Fitness initialized to 100
    const out = forwardBatch(positions, targetIndices, targets, directions, fitnesses);
    
}

main();

export { forwardBatch }