import * as tf from '@tensorflow/tfjs';
import { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';
import '@tensorflow/tfjs-backend-webgl';
import { getSubsetsOfSizeK } from './util';
import { settings } from '../settings';

//const PLANE_INDICES = getPlaneIndices(C);

function normalize2D(A: tf.Tensor2D, dim: number): tf.Tensor2D {
    const norm = tf.norm(A, 'euclidean', dim);
    const normalized = A.div<tf.Tensor2D>(norm.expandDims(-1).tile([1, A.shape[1]]));
    return normalized
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
    const T = settings["Number of Food"];
    const C = settings["Number of Dimensions"];
    const dot1 = tf.matMul(U, V, false, true);
    const dot2 = V.square().sum(-1).expandDims(-1).tile([1, 1, T]).transpose([0, 2, 1]) // (B, T, C)
    const div1 = tf.div(dot1, dot2).transpose([0, 2, 1]).expandDims(-1).tile([1, 1, 1, C]);
    const dot3 = V.expandDims(-2).tile([1, 1, T, 1]);
    const mul1 = tf.mul(div1, dot3).transpose([0, 2, 1, 3]);
    return mul1 as tf.Tensor4D
});

const distanceFromPlaneBatch = (planePoint: tf.Tensor3D, normalPoint: tf.Tensor3D, position: tf.Tensor2D, velocity: tf.Tensor3D): [tf.Tensor2D, tf.Tensor1D, tf.Tensor3D] => tf.tidy(() => {
    const C = settings["Number of Dimensions"];
    const B = settings["Batch Size"];
    // Offset the plane to ensure accurate distance
    const offsetPoint = tf.sub<tf.Tensor3D>(planePoint, position.expandDims(-2).tile([1, planePoint.shape[1], 1])); // (B, 2 * C, C)
    // Find the point that the line from the velocity vector intersects the plane
    const intersectionPoint = projectOntoPlaneBatch(offsetPoint, velocity, normalPoint) as tf.Tensor3D;
    // Calculate the distance between the point on the plane and the velocity vector
    const distance = tf.add<tf.Tensor3D>(intersectionPoint, position.expandDims(1).tile([1, C * 2, 1]).neg()).square().sum<tf.Tensor2D>(2);
    const relativeIntersectionPoint = tf.add<tf.Tensor3D>(intersectionPoint, position.expandDims(1).tile([1, C * 2, 1]).neg())
    const relativeIntersectionPointPositive = relativeIntersectionPoint.slice([0, 0, 0], [B, C, C])
    const relativeIntersectionPointNegative = relativeIntersectionPoint.slice([0, C, 0], [B, C, C])
    const dot = tf.mul(relativeIntersectionPointNegative, relativeIntersectionPointPositive).sum(2); // (B, C)
    const dotNorm = tf.mul(tf.norm(relativeIntersectionPointNegative, 'euclidean', 2), tf.norm(relativeIntersectionPointPositive, 'euclidean', 2)); // (B, C)
    const similarityOpposite = tf.divNoNan(dot, dotNorm).less(0).slice([0, 0], [B, 1]).squeeze([1]) as tf.Tensor1D; // (B, C)
    return [distance, similarityOpposite, intersectionPoint]
});

function dotProduct<T extends tf.Tensor>(A: T, B: T, axis?: number, keepDims?: boolean): T {
    if ((axis || 0) >= A.rank || (axis || 0) >= B.rank) {
        throw new Error("Dimension must be less than rank of tensor");
    }

    return tf.tidy(() => tf.mul(A, B).sum(axis, keepDims));
}

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
    const xor = tf.logicalXor<tf.Tensor2D>(firstHalf.less(0), secondHalf.less(0)).logicalOr<tf.Tensor2D>(firstHalf.equal(tf.fill([B, C], Infinity))).tile<tf.Tensor2D>([1, 2]);
    return xor
})

const calculateNearbyBounds = (position: tf.Tensor2D, direction: tf.Tensor3D, scale: number) => tf.tidy(() => {
    const raycasted = raycast(position, direction, scale);
    const mask = inBounds(raycasted).cast<tf.Tensor2D>('float32');
    const distance = closestDistance(raycasted.mul(mask));
    
    return distance
})

const projectDirectionToBounds = (position: tf.Tensor2D, direction: tf.Tensor3D, scale: number): tf.Tensor2D => tf.tidy(() => {
    const B = position.shape[0];
    const C = position.shape[1];
    const distance = calculateNearbyBounds(position, direction, scale).expandDims(-1).tile([1, C]) as tf.Tensor2D;
    
    const forwardVectors = direction.slice([0, 0, 0], [B, 1, C]).squeeze([1]) as tf.Tensor2D;
    const proj = tf.add(position, tf.mul(forwardVectors, distance)) as tf.Tensor2D;
    
    return proj
})

const calculateNearbyBody = (position: tf.Tensor2D, direction: tf.Tensor3D, history: tf.Tensor3D, sensitivity: number, range: number) => tf.tidy(() => {
    // position: (B, C)
    // history: (B, T, C) <- T = global max food
    const T = settings["Number of Food"];
    const C = settings["Number of Dimensions"];
    const B = settings["Batch Size"];

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
    const C = settings["Number of Dimensions"];

    // Translate vector space, positioning position at the origin
    const targetRelativePosition = target.add<tf.Tensor2D>(position.neg()); // (B, C)
    const targetBroadcasted: tf.Tensor3D = targetRelativePosition.expandDims(1).tile([1, C, 1]); // (B, C, C)

    // Calculate cosine similarity between normalized targets and direction vectors
    const dot = tf.mul(direction, targetBroadcasted).sum(2); // (B, C)
    const dotNorm = tf.mul(tf.norm(direction, 'euclidean', 2), tf.norm(targetBroadcasted, 'euclidean', 2)); // (B, C)
    const similarity = tf.divNoNan(dot, dotNorm); // (B, C)
    
    return similarity
});

const generateWeights = (weightsA: tf.LayerVariable[], weightsB: tf.LayerVariable[], crossoverProbability: number, differentialWeight: number): void => {
    weightsB.map((weights: tf.LayerVariable, index: number) => tf.tidy(() => {
        const B = (weights.read() as tf.Tensor3D).shape[0];
        const T = (weights.read() as tf.Tensor3D).shape[1];
        const C = (weights.read() as tf.Tensor3D).shape[2];
        
        const ranges1 = tf.range(0, B, ...[,], "int32").expandDims().tile([B, 1]);
        const ranges2 = tf.range(0, B, ...[,], "int32").expandDims().tile([B, 1]).transpose();
        const upper = tf.greater(ranges1, ranges2).cast('int32').mul(B).reverse(0);
        const idx = tf.add(ranges1, ranges2).sub(upper).slice([0, 0], [B, B-1]);
        const indices = Array.from(Array(B-1).keys(), (v: number) => v);
        tf.util.shuffle(indices)
        const randomIndices = tf.tensor(indices, ...[,], "int32").slice([0], [3]).expandDims().tile([B, 1]);

        const idx2 = tf.gather(idx, randomIndices, 1, 1).cast("int32"); // (B, 3)
        const uniqueWeights = tf.gather(weights.read(), idx2) as unknown as tf.Tensor4D; // (B, 3, T, C)
        const rValues = tf.randomUniform(weights.read().shape, ...[,,], "float32") // (B, T, C)
        const mask = rValues.greater(crossoverProbability).cast('int32'); // (B, T, C)
        let positiveMask = mask.clipByValue(0, 1);
        if (C != 1) {
            const mask2 = tf.oneHot(tf.randomUniformInt([B, T], 0, C), C).cast('int32'); // (B, T, C)
            positiveMask = mask.add(mask2).clipByValue(0, 1);
        }

        const negativeMask = positiveMask!.notEqual(1);
        
        const p0 = tf.slice(uniqueWeights, [0, 0, 0, 0], [B, 1, T, C]).squeeze<tf.Tensor3D>([1]);
        const p1 = tf.slice(uniqueWeights, [0, 1, 0, 0], [B, 1, T, C]).squeeze<tf.Tensor3D>([1]);
        const p2 = tf.slice(uniqueWeights, [0, 2, 0, 0], [B, 1, T, C]).squeeze<tf.Tensor3D>([1]);

        const updateWeightRule = p1.sub<tf.Tensor3D>(p2).mul<tf.Tensor3D>(differentialWeight).add<tf.Tensor3D>(p0);
        const dontUpdateWeightRule = negativeMask.mul<tf.Tensor3D>(weights.read());
        const transformedWeights = tf.keep(updateWeightRule.mul<tf.Tensor3D>(positiveMask).add<tf.Tensor3D>(dontUpdateWeightRule));
        weightsA[index].write(tf.keep(transformedWeights))
    }));
}



const compareWeights = (fitnessesA: tf.Tensor1D, weightsA: tf.LayerVariable[], fitnessesB: tf.Tensor1D, weightsB: tf.LayerVariable[]): void => tf.tidy(() => {
    const fitnessesStacked: tf.Tensor2D = tf.stack<tf.Tensor1D>([fitnessesA, fitnessesB], 1) as tf.Tensor2D;
    const argMaxFitnesses = tf.argMax(fitnessesStacked, 1);

    
    for (let weightsNum = 0; weightsNum < weightsA.length; weightsNum++) {
        const weightsStacked: tf.Tensor4D = tf.stack<tf.Tensor3D>([weightsA[weightsNum].read() as tf.Tensor3D, weightsB[weightsNum].read() as tf.Tensor3D], 1) as tf.Tensor4D;
        const fittestWeightsPartial: tf.Tensor3D = tf.gather(weightsStacked, argMaxFitnesses, 1, 1) as unknown as tf.Tensor3D;
        weightsA[weightsNum].write(tf.keep(fittestWeightsPartial));
        fittestWeightsPartial.dispose();
        weightsStacked.dispose();
    }
});

interface DEModelInputArray {
    position: number[][];
    direction: number[][][];
    fitness: number[];
    targetIndices: number[];
    history: number[][][];
    alive: number[];
}

const batchArraySync = (a: DEModelInput): DEModelInputArray => tf.tidy(() => {
    return {
        position: a[0].arraySync() as number[][],
        direction: a[1].arraySync() as number[][][],
        fitness: a[2].arraySync() as number[],
        targetIndices: a[3].arraySync() as number[],
        history: a[4].arraySync() as number[][][],
        alive: a[5].arraySync() as number[]
    }
});

type DEModelInput = [tf.Tensor2D, tf.Tensor3D, tf.Tensor1D, tf.Tensor1D, tf.Tensor3D, tf.Tensor1D];
class DEModel extends tf.layers.Layer {
    public inputShape: tf.Shape;
    public B: number;
    public C: number;
    public T: number;
    public H: number;
    public dtype: tf.DataType;
    public wInputHidden: undefined | tf.LayerVariable;
    public bInputHidden: undefined | tf.LayerVariable;
    public wHiddenOutput: undefined | tf.LayerVariable;
    public bHiddenOutput: undefined | tf.LayerVariable;
    public wTarget: undefined | tf.LayerVariable;
    public ttl: number;
    public planeIndices: tf.Tensor2D;
    public gameSize: number;

    constructor(config: LayerArgs) {
        super(config);
        this.inputShape = (config.batchInputShape || config.inputShape)!
        if (this.inputShape.length != 3) {
            throw new Error("Input shape must specify 3 dimensions");
        }
        this.B = this.inputShape[0]!;
        this.T = this.inputShape[1]!;
        this.C = this.inputShape[2]!;
        this.H = settings["Number of Hidden Layer Nodes"];
        this.ttl = settings["Time To Live"];
        this.planeIndices = this.#generatePlaneIndices;
        this.gameSize = settings["Bound Box Length"];
        this.dtype = config.dtype || "float32";
    }

    get #generatePlaneIndices(): tf.Tensor2D {
        const _tensorList: tf.Tensor1D[] = getSubsetsOfSizeK<number>(Array.from(Array(this.C).keys()), 2)
            .map((v: number[]) => tf.tensor(v, ...[,], 'int32'));
        return tf.stack<tf.Tensor1D>(_tensorList) as tf.Tensor2D;
    }

    get inputLayerSize() {
        return this.C * 3 + 1;
    }

    get outputLayerSize() {
        return ((this.C - 1) * this.C) + 1
    }

    build() {
        this.wInputHidden = this.addWeight('wInputHidden', [this.B, this.H, this.inputLayerSize], this.dtype, tf.initializers.randomUniform({ minval: -1, maxval: 1 }))
        this.bInputHidden = this.addWeight('bInputHidden', [this.B, this.H, 1], this.dtype, tf.initializers.randomUniform({ minval: -1, maxval: 1 }))
        this.wHiddenOutput = this.addWeight('wHiddenOutput', [this.B, this.outputLayerSize, this.H], this.dtype, tf.initializers.randomUniform({ minval: -1, maxval: 1 }))
        this.bHiddenOutput = this.addWeight('bHiddenOutput', [this.B, this.outputLayerSize, 1], this.dtype, tf.initializers.randomUniform({ minval: -1, maxval: 1 }))
        this.wTarget = this.addWeight('wTargets', [this.T, this.C], this.dtype, tf.initializers.randomUniform({ minval: -this.gameSize, maxval: this.gameSize }), ...[,], false);
    }

    buildFrom(other: DEModel, crossoverProbability: number, differentialWeight: number) {
        generateWeights(this.trainableWeights, other.trainableWeights, crossoverProbability, differentialWeight);
        this.wTarget!.write(other.wTarget!.read());
    }

    buildFromCompare(inputs: DEModelInput, otherInputs: DEModelInput, other: DEModel)  {
        compareWeights(inputs[2], this.trainableWeights, otherInputs[2], other.trainableWeights);
    }

    call(inputs: DEModelInput): DEModelInput {
        if (!this.wInputHidden) throw new Error("Weights not initialized")
        if (!this.wHiddenOutput) throw new Error("Weights not initialized")
        if (!this.wTarget) throw new Error("Weights not initialized")
        
        const out: DEModelInput = tf.tidy(() => {
            const indexedFood = (this.wTarget!.read() as tf.Tensor2D).gather(inputs[3]);
            const sequentialInput = this.#sensoryData(inputs, indexedFood);
            const [sequentialOutput1, sequentialOutput2] = this.#sequentialForward(sequentialInput.expandDims(-1) as tf.Tensor3D);
            const [position, direction] = this.#movementForward(inputs[0], inputs[1], sequentialOutput1, sequentialOutput2);
            const [fitness, targetIndices, alive] = this.#logicForward(inputs, position, indexedFood, sequentialInput);
            const history = this.#historyForward(inputs, position, targetIndices);
            
            tf.dispose(inputs);
            
            return [tf.keep(position), tf.keep(direction), tf.keep(fitness), tf.keep(targetIndices), tf.keep(history), tf.keep(alive)];
        })
        return out;
    }

    get resetInput(): DEModelInput {
        return [
            tf.keep(tf.zeros([this.B, this.C])),
            tf.keep(tf.eye(this.C).expandDims(0).tile([this.B, 1, 1])),
            tf.keep(tf.ones([this.B]).mul(this.ttl) as tf.Tensor1D),
            tf.keep(tf.zeros([this.B], 'int32')),
            tf.keep(tf.zeros([this.B, this.T, this.C - 1]).concat(tf.range(0, this.T).reshape([1, this.T]).tile([this.B, 1]).expandDims(-1).neg(), -1) as tf.Tensor3D),
            tf.keep(tf.ones([this.B], 'int32'))
        ]
    }

    #sensoryData(inputs: DEModelInput, indexedFood: tf.Tensor2D): tf.Tensor2D {
        const out = tf.tidy(() => {
            // Food data processing and vectorizing (n_dims)
            const nearbyFood = calculateNearbyTarget(inputs[0], inputs[1], indexedFood); // (B, C)
            // Body data processing and vectorizing (5) (n_dims * 2 - 1)
            const nearbyBody = calculateNearbyBody(inputs[0], inputs[1], inputs[4], 0.95, 1000); // (B, C * 2)
            // Bounds data processing and vectorizing (3) (n_dims * 2 - 1)
            const nearbyBounds = calculateNearbyBounds(inputs[0], inputs[1], this.gameSize).expandDims(-1).clipByValue(0, 1200); // (B, 1)
            
            // Concatenate all sensory data to a single input vector
            return tf.concat([nearbyFood, nearbyBounds, nearbyBody], 1) as tf.Tensor2D; // (B, C * 5)
        })
        return out
    }

    // Feed forward linear layers (no bias)
    #sequentialForward(input: tf.Tensor3D): [tf.Tensor2D, tf.Tensor1D] {
        if (!this.wInputHidden) throw new Error("Weights not initialized")
        if (!this.wHiddenOutput) throw new Error("Weights not initialized")
        if (!this.wTarget) throw new Error("Weights not initialized")
        const [out1, out2] = tf.tidy(() => {
            const hidden = tf.sigmoid(tf.matMul(this.wInputHidden!.read(), input, false, false).add(this.bInputHidden!.read())).mul(2).sub(1);
            const logits = tf.sigmoid(tf.matMul(this.wHiddenOutput!.read(), hidden, false, false).add(this.bHiddenOutput!.read())).mul(2).sub(1);
            const control = logits.squeeze([2]).slice([0, ((this.C - 1) * this.C)], [this.B, 1]).squeeze([1]).clipByValue(0, 1) as tf.Tensor1D;
            const softmax = tf.softmax<tf.Tensor2D>(logits.squeeze([2]).slice([0, 0], [this.B, ((this.C - 1) * this.C)]) as tf.Tensor2D, 1); // (B, (((C-1)*C)/2))
            return [softmax, control]
        });
        input.dispose();
        return [out1, out2];
    }

    #movementForward(position: tf.Tensor2D, direction: tf.Tensor3D, input1: tf.Tensor2D, input2: tf.Tensor1D): [tf.Tensor2D, tf.Tensor3D] {
        const [ out1, out2 ] = tf.tidy(() => {
            
            
            const planeIndices = this.planeIndices.reverse(1).concat(this.planeIndices, 0);
            const indices = planeIndices.gather(input1.argMax(1));
            
            const nextDirections = rotateBatch(direction, indices, input2.mul(Math.PI * 0.5));
            const nextVelocity: tf.Tensor2D = nextDirections.slice([0, 0], [this.B, 1]).squeeze([1]);
            const nextPositions: tf.Tensor2D = position.add(nextVelocity.mul(1));
            return [nextPositions, nextDirections ];
        })
        return [out1, out2];
    }

    #logicForward(inputs: DEModelInput, nextPosition: tf.Tensor2D, indexedFood: tf.Tensor2D, sensoryData: tf.Tensor2D): [ tf.Tensor1D, tf.Tensor1D, tf.Tensor1D ] {
        const [ out1, out2, out3 ] = tf.tidy(() => {
            const outOfBounds = sensoryData.slice([0, this.inputLayerSize - 1], [this.B, 1]).squeeze();
            
            const distanceAfter = tf.squaredDifference(nextPosition, indexedFood).sum(1).sqrt(); // Euclidean distance
            const distanceBefore = tf.squaredDifference(inputs[0], indexedFood).sum(1).sqrt();
            const distanceDifferences = tf.sub(distanceBefore, distanceAfter);

            const touchingFood = distanceAfter.lessEqual(3).cast("int32");
            const improvingMask = distanceDifferences.greater(1).mul(1).add(touchingFood.mul(this.ttl));
            const degradingMask = distanceDifferences.lessEqual(1).mul(-5);

            const isAliveMask = inputs[2].greater(inputs[3].mul(this.ttl)).cast('int32').mul(inputs[5]).mul(outOfBounds) as tf.Tensor1D;
            const nextFitnesses: tf.Tensor1D = inputs[2].add(improvingMask.add(degradingMask).mul(isAliveMask));
            const nextTargetIndices = inputs[3].add(touchingFood) as tf.Tensor1D;
            return [ nextFitnesses, nextTargetIndices, isAliveMask ];
        });
        return [ out1, out2, out3 ];
    }

    #historyForward(inputs: DEModelInput, nextPosition: tf.Tensor2D, nextTargetIndices: tf.Tensor1D): tf.Tensor3D {
        const S = settings["Starting Snake Length"];
        const out = tf.tidy(() => {
            const cutoffMask = tf.range(0, this.T).expandDims().tile([this.B, 1]).less(nextTargetIndices.expandDims(-1).tile([1, this.T]).add(S)).cast('int32'); // (B, T)
            const cutoffKeep = tf.concat([nextPosition.expandDims(1), inputs[4]], 1).slice([0, 0, 0], [this.B, this.T, this.C]); // (B, T, C)
            const nextHistory = cutoffKeep.div(cutoffMask.expandDims(1).tile([1, this.C, 1]).transpose([0, 2, 1])) as tf.Tensor3D; // (B, T, C)
            return nextHistory;
        })
        return out;
    }

    getConfig() {
        const config = super.getConfig();
        Object.assign(config, {
            B: this.B,
            T: this.T,
            C: this.C
        });
        return config;
    }

    static get className() {
        return 'DEModel';
    }
}

export { batchArraySync, compareWeights, generateWeights, DEModel, calculateNearbyBounds, projectDirectionToBounds }
export type { DEModelInputArray, DEModelInput }