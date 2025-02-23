import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';


const PLANE_INDICES = getPlaneIndices(5);
const TTL = 100;

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
    
    const basisVectors: tf.Tensor4D = tf.gather<tf.Tensor3D>(_normalVectors, planeIndices, 1, 1) as unknown as tf.Tensor4D; // (B, (((C-1)*C)/2), T, C)
    const basisVectorsTransposed = basisVectors.transpose([0, 1, basisVectors.rank - 1, basisVectors.rank - 2]); // (B, (((C-1)*C)/2), C, T)
    const basisReshaped = basisVectorsTransposed.reshape([B * (((C-1)*C)/2), C, T]);
    
    //const [q, r] = tf.linalg.qr(tf.concat([basisVectorsTransposed, tf.zeros([B, (((C-1)*C)/2), C, 1])], -1));
    
    //const orthogonalUnitVectors = tf.slice(q, [0, 0, 0, 0], [B, (((C-1)*C)/2), C, T]);
    const _vProjected: tf.Tensor4D = tf.matMul(basisVectors, _normalVectors.expandDims(1).tile([1, (((C-1)*C)/2), 1, 1])); // (B, (((C-1)*C)/2), T, C)
    const BatchCosine: tf.Tensor2D = tf.cos<tf.Tensor2D>(angles); // (B, (((C-1)*C)/2))
    const BatchSine: tf.Tensor2D = tf.sin<tf.Tensor2D>(angles); // (B, (((C-1)*C)/2))
    const _rotationMatrixIdentity: tf.Tensor4D = tf.eye(T).expandDims(0).tile([(((C-1)*C)/2), 1, 1]).expandDims(0).tile([B, 1, 1, 1]); // (B, (((C-1)*C)/2), T, T)
    const _range: tf.Tensor3D = tf.range(0, (((C-1)*C)/2), ...[,], "int32").expandDims(0).tile([4, 1]).transpose().reshape([1, (((C-1)*C)/2) * 4]).transpose().expandDims(0).tile([B, 1, 1]) // (B, (((C-1)*C)/2)*4, 1)
    const _rangeBatch = tf.range(0, B, ...[,],"int32").expandDims(-1).tile([1, (((C-1)*C)/2)*4]).expandDims(-1)
    const _indices: tf.Tensor3D = tf.tensor2d([[0, 0], [0, 1], [1, 0], [1, 1]], ...[,], "int32").expandDims().tile([B, (((C-1)*C)/2), 1]);
    const _repeatedIndices = tf.concat([_range, _indices], 2);
    
    const _repeatedIndicesBatch = tf.concat([_rangeBatch, _repeatedIndices], 2).reshape([B*(((C-1)*C)/2)*4, 4]);
    const _rot: tf.Tensor2D = tf.stack<tf.Tensor2D>([BatchCosine, BatchSine.neg(), BatchSine, BatchCosine], 1).transpose([0, 2, 1]).reshape([B, 1, _rotationMatrixIdentity.shape[1] * 4]).squeeze([1]).reshape([B*(((C-1)*C)/2)*4]);
    const _rotationMatrix = tf.tensorScatterUpdate(_rotationMatrixIdentity, _repeatedIndicesBatch, _rot);
    const _vRotatedProjected = tf.matMul(_rotationMatrix, _vProjected);
    const _vRotated = tf.matMul(basisVectors.transpose([0, 1, 3, 2]), _vRotatedProjected);
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


function feedForward(weights: tf.Tensor[], biases?: tf.Tensor[], bias: boolean = true) {
    if (biases == undefined && bias == true) {
        throw new Error("Bias is expected");
    }
    if (biases != undefined && bias == false) {
        throw new Error("Bias is not expected");
    }
    if (biases?.length != weights.length) {
        throw new Error("Weights and biases lengths must match");
    }
    function forward(weight: tf.Tensor[], bias?: tf.Tensor1D[] | tf.Tensor1D) {
        if (!weight) {

        }

    }

}

interface CustomModelLayerConfig {
    B: number; // The batch size
    C: number; // The channel size (number of spatial dimensions)
}
class FoodDistanceLayer extends tf.layers.Layer {
    B: number;
    C: number;

    constructor(tfConfig: LayerArgs, customConfig: CustomModelLayerConfig) {
        super(tfConfig);
        this.B = customConfig.B;
        this.C = customConfig.C;
    }

    call(inputs: [tf.Tensor2D, tf.Tensor3D, tf.Tensor2D]): tf.Tensor2D {
        // inputs[0]: positions (B, C)
        // inputs[1]: directions (B, C, C)
        // inputs[2]: indexedFood (B, C)

        // Extend basis vectors to negative axis
        const signedVelocity: tf.Tensor3D = tf.concat<tf.Tensor3D>([inputs[1], inputs[1].mul<tf.Tensor3D>(-1)], 1); // (B, 2 * C, C)
        // Extend unit vectors to negative axis
        const signedUnitVectors: tf.Tensor3D = tf.concat([tf.eye(this.C), tf.eye(this.C).mul(-1)]).expandDims().tile([this.B, 1, 1]); // (B, 2 * C, C)
        // Food data processing and vectorizing (n_dims * 2)
        const foodNormals: tf.Tensor2D = cosineSimilarity(inputs[0].sub(inputs[2]).expandDims(1).tile([1, this.C * 2, 1]), signedVelocity, 2) as tf.Tensor2D; // (B, 2 * C)
        const nearbyFood: tf.Tensor2D = foodNormals.clipByValue(-1, 1) as tf.Tensor2D; // (B, 2 * C)
        return nearbyFood
    }

    getConfig() {
        const config = super.getConfig();
        Object.assign(config, {B: this.B, C: this.C});
        return config
    }

    static get className() {
        return 'FoodDistanceLayer'
    }
}

tf.serialization.registerClass(FoodDistanceLayer);

class NearbyBoundsLayer extends tf.layers.Layer {
    B: number;
    C: number;

    constructor(tfConfig: LayerArgs, customConfig: CustomModelLayerConfig) {
        super(tfConfig);
        this.B = customConfig.B;
        this.C = customConfig.C;
    }

    call(inputs: [tf.Tensor2D, tf.Tensor3D]): tf.Tensor2D {
        return tf.tidy(() => {
            const identity = tf.eye(this.C);
            const extendedIdentity = tf.concat([identity, identity.neg()]).mul<tf.Tensor2D>(10).expandDims().tile<tf.Tensor3D>([this.B, 1, 1]); // (B, 2 * C, C)
            const extendedVelocity = tf.concat([inputs[1], inputs[1].neg()], 1); // (B, 2 * C, C)
            const nearbyBounds: tf.Tensor2D = distanceFromPlaneBatch(extendedIdentity, extendedIdentity.neg(), inputs[0], extendedVelocity).softmax<tf.Tensor2D>(1) // (B, 2 * C)
            return nearbyBounds
        })
    }
}



const forwardBatch = (positions: tf.Tensor2D, targetIndices: tf.Tensor1D, targetPositions: tf.Tensor2D, directions: tf.Tensor3D, fitnesses: tf.Tensor1D, weights: tf.Tensor3D[]) => tf.tidy(() => {
    const B = positions.shape[0]; 
    const C = positions.shape[1];
    
    
    const indexedFood = targetPositions.gather(targetIndices);

    // Extend basis vectors to negative axis
    const signedVelocity: tf.Tensor3D = tf.concat<tf.Tensor3D>([directions, directions.mul<tf.Tensor3D>(-1)], 1); // (B, 2 * C, C)

    // Extend unit vectors to negative axis
    const signedUnitVectors: tf.Tensor3D = tf.concat([tf.eye(C), tf.eye(C).mul(-1)]).expandDims().tile([B, 1, 1]); // (B, 2 * C, C)

    // Food data processing and vectorizing (n_dims * 2)
    const foodNormals = cosineSimilarity(positions.sub(indexedFood).expandDims(1).tile([1, C * 2, 1]), signedVelocity, 2); // (B, 2 * C)
    const nearbyFood = foodNormals.clipByValue(-1, 1);
    const foodDistance = tf.squaredDifference(positions, indexedFood).sum(-1).sqrt(); // (B,)

    // Body data processing and vectorizing (5) (n_dims * 2 - 1)
    /*const bodyNormals = signedVelocity.add(positions.expandDims(0).tile([this.numberOfDimensions * 2, 1]));
    const snakeHistoryHeadOmitted = snakeHistory.slice([0, 0], [snakeHistory.shape[0] - 1, this.numberOfDimensions]); // Omit the head of the snake
    const bodyDistances = cdist(bodyNormals, snakeHistoryHeadOmitted);
    const nearbyBody = tf.any(tf.concat([bodyDistances.slice([0], [this.numberOfDimensions]), bodyDistances.slice([this.numberOfDimensions + 1], [this.numberOfDimensions - 1])]).equal(0), 1).cast('float32');*/

    // Bounds data processing and vectorizing (3) (n_dims * 2 - 1)
    // TODO: Add new raycasting function to determine bounds
    const identity = tf.eye(C);
    const extendedIdentity = tf.concat([identity, identity.neg()]).mul<tf.Tensor2D>(10).expandDims().tile<tf.Tensor3D>([B, 1, 1]); // (B, 2 * C, C)
    const extendedVelocity = tf.concat([directions, directions.neg()], 1); // (B, 2 * C, C)
    const nearbyBounds: tf.Tensor2D = distanceFromPlaneBatch(extendedIdentity, extendedIdentity.neg(), positions, extendedVelocity).softmax<tf.Tensor2D>(1) // (B, 2 * C)
    const inputs = tf.concat([nearbyFood, nearbyBounds], 1).expandDims(-1); // (B, 4 * C)
    
    // Feed forward linear layers (no bias)
    const hidden = tf.matMul(weights[0], inputs, false, false)
    const hiddenReLU = tf.clipByValue(hidden, 0, 1); // ReLU activation function
    const logits = tf.matMul(weights[1], hidden, false, false)
    const logitsReLU = tf.clipByValue(logits, 0, 1); // ReLU activation function
    const logitsFlattened = logits.squeeze([2]) as tf.Tensor2D; // (B, (((C-1)*C)/2))
    
    
    const angles = normalize2D(logitsFlattened, 1).sub(0.5).mul<tf.Tensor2D>(Math.PI); // (B, (((C-1)*C)/2))
    console.log(angles.arraySync())
    const nextDirections = rotateBatch(directions, PLANE_INDICES.expandDims(0).tile([B, 1, 1]), angles);
    
    const nextVelocity: tf.Tensor2D = nextDirections.slice([0, 0], [B, 1]).squeeze([1]);
    const nextPositions: tf.Tensor2D = positions.add(nextVelocity);
    const nextDistances = tf.squaredDifference(nextPositions, indexedFood).sum(1).sqrt(); // Euclidean distance
    const previousDistances = tf.squaredDifference(positions, indexedFood).sum(1).sqrt();
    const distanceDifferences = tf.sub(previousDistances, nextDistances);
    
    const touchingFood = nextDistances.lessEqual(3).cast("int32");
    const improvingMask = distanceDifferences.greater(1).mul(1).add(touchingFood.mul(TTL));
    const degradingMask = distanceDifferences.lessEqual(1).mul(-5);
    
    const isAliveMask = fitnesses.greater(targetIndices.mul(TTL)).cast('int32');
    const isDeadMask = fitnesses.lessEqual(targetIndices.mul(TTL)).cast('int32');
    const nextFitnesses: tf.Tensor1D = fitnesses.add(improvingMask.add(degradingMask).mul(isAliveMask)).add(tf.mul(TTL, isDeadMask));
    const nextTargetIndices = targetIndices.add(touchingFood);
    
    return [ nextPositions, nextDirections, nextFitnesses, nextTargetIndices ]
});

const generateWeights = (weightsA: tf.Tensor3D[], crossoverProbability: number, differentialWeight: number): tf.Tensor3D[] => tf.tidy(() => {
    const B = weightsA[0].shape[0];
    
    let weightsB: tf.Tensor3D[] = []
    for (let weightsNum = 0; weightsNum < weightsA.length; weightsNum++) {
        const T = weightsA[weightsNum].shape[1];
        const C = weightsA[weightsNum].shape[2];
        
        const ranges1 = tf.range(0, B, ...[,], "int32").expandDims().tile([B, 1]);
        const ranges2 = tf.range(0, B, ...[,], "int32").expandDims().tile([B, 1]).transpose();
        const upper = tf.greater(ranges1, ranges2).cast('int32').mul(B).reverse(0);
        const idx = tf.add(ranges1, ranges2).sub(upper).slice([0, 1], [B, B-1]);
        let indices = Array.from(Array(B-1).keys(), (v: number) => v+1);
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
        const transformedWeights = updateWeightRule.mul<tf.Tensor3D>(positiveMask).add<tf.Tensor3D>(dontUpdateWeightRule);

        weightsB.push(transformedWeights);
    }

    return weightsB
});



function compareWeights(fitnessesA: tf.Tensor1D, weightsA: tf.Tensor3D[], fitnessesB: tf.Tensor1D, weightsB: tf.Tensor3D[]): tf.Tensor3D[] {
    const B = fitnessesA.shape[0];
    
    const fitnessesStacked: tf.Tensor2D = tf.stack<tf.Tensor1D>([fitnessesA, fitnessesB], 1) as tf.Tensor2D;
    const argMaxFitnesses = tf.argMax(fitnessesStacked, 1);

    let weightsC = [];
    for (let weightsNum = 0; weightsNum < weightsA.length; weightsNum++) {
        const weightsStacked: tf.Tensor4D = tf.stack<tf.Tensor3D>([weightsA[weightsNum], weightsB[weightsNum]], 1) as tf.Tensor4D;
        const fittestWeightsPartial: tf.Tensor3D = tf.gather(weightsStacked, argMaxFitnesses, 1, 1) as unknown as tf.Tensor3D;
        weightsC.push(fittestWeightsPartial);
    }

    return weightsC
}

function unitTest() {
    
    //tf.enableDebugMode();
    const weights = tf.tidy(() => {
        // Model unit test for minimizeBatch()
        const TTL = 100;
        const B = 100; 
        const C = 5; // Number of dimensions
        const G = 4; // Number of food
        const T = 80;
        
        
        const fitnesses: tf.Tensor1D = tf.ones([B]).mul(TTL);
        const positions: tf.Tensor2D = tf.zeros([B, C]);
        const targetIndices: tf.Tensor1D = tf.zeros([B], 'int32');
        const targets: tf.Tensor2D = tf.randomUniform([100, C], -10, 10);
        const directions: tf.Tensor3D = tf.eye(C).expandDims(0).tile([B, 1, 1]);
        
        
        let weightsA: tf.Tensor3D[] = ([tf.randomUniform([100, T, 4 * C]), tf.randomUniform([100, Math.floor((C - 1) * C / 2), T])]);
        for (let i = 0; i < 5; i++) {
            
            const weightsB = generateWeights(weightsA, 0.7, 0.6);
            let [ nextPositionsA, nextDirectionsA, nextFitnessesA, nextTargetIndicesA ] = forwardBatch(positions, targetIndices, targets, directions, fitnesses, weightsA);
            let [ nextPositionsB, nextDirectionsB, nextFitnessesB, nextTargetIndicesB ] = forwardBatch(positions, targetIndices, targets, directions, fitnesses, weightsB);
            
            for (let j = 0; j < 25; j++) {
                if (tf.getBackend() === 'cpu') {
                    throw new Error("CPU backend is not supported");
                }
                
                console.time("forwardBatch");
                [ nextPositionsA, nextDirectionsA, nextFitnessesA, nextTargetIndicesA ] = forwardBatch(nextPositionsA as tf.Tensor2D, nextTargetIndicesA as tf.Tensor1D, targets, directions, nextFitnessesA as tf.Tensor1D, weightsA);
                [ nextPositionsB, nextDirectionsB, nextFitnessesB, nextTargetIndicesB ] = forwardBatch(nextPositionsB as tf.Tensor2D, nextTargetIndicesB as tf.Tensor1D, targets, directions, nextFitnessesB as tf.Tensor1D, weightsB);
                //console.log(nextPositionsA.arraySync()[0], nextFitnessesA.arraySync(), targets.gather(nextTargetIndicesA).arraySync());
                console.timeEnd("forwardBatch");
            }
            weightsA = compareWeights(nextFitnessesA as tf.Tensor1D, weightsA, nextFitnessesB as tf.Tensor1D, weightsB);
        }
        return weightsA

    });
    console.log(weights);
    weights[0].dispose();
    weights[1].dispose();
    tf.dispose(weights);


    

    



    
    
    
    
}
export { forwardBatch, unitTest }