import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import Renderer from './renderer';
import { forwardBatch } from './modelV2';


const TTL = 100;
const INITIAL_SCORE = 5;

const INPUT_LAYER_SIZE = 10;  // Example size
const HIDDEN_LAYER_SIZE = 32;
const OUTPUT_LAYER_SIZE = 5;

const BOUNDS_SCALE = 20;

function loadMesh(mesh: THREE.Mesh<THREE.BoxGeometry, THREE.MeshBasicMaterial, THREE.Object3DEventMap>, position: number[]) {
    mesh.position.set(position[0], position[1], position.length > 2 ? position[2] : 0)
}

// Computes batched the p-norm distance between each pair of the two collections of row vectors
// Adapted from Pytorch method here: https://pytorch.org/docs/stable/generated/torch.cdist.html
function cdist(A: tf.Tensor, B: tf.Tensor): tf.Tensor {
    const expandedA = tf.expandDims(A, 1);
    const expandedB = tf.expandDims(B, 0);
    const difference = tf.sub(expandedA, expandedB);
    const squaredDifference = tf.pow(difference, 2);
    const sumSquaredDifference = tf.sum(squaredDifference, 2);
    const distance = tf.sqrt(sumSquaredDifference);
    return distance;
}

function cosineSimilarity(A: tf.Tensor, B: tf.Tensor, dim: number): tf.Tensor {
    const euclideanNormA = tf.square(A).sum(dim).sqrt();
    const euclideanNormB = tf.square(B).sum(dim).sqrt();
    const productTop = tf.mul(A, B).sum(dim);
    const productBottom = tf.mul(euclideanNormA, euclideanNormB);
    const diff = tf.divNoNan(productTop, productBottom);
    return diff
}

function normalize<T extends tf.Tensor<tf.Rank>>(A: tf.Tensor, dim: number): T {
    const norm = tf.norm(A, 'euclidean', dim);
    const normalized = A.div<T>(norm);
    return normalized
}

function shuffle(array: number[]) {
    let currentIndex = array.length;
  
    // While there remain elements to shuffle...
    while (currentIndex != 0) {
  
      // Pick a remaining element...
      let randomIndex = Math.floor(Math.random() * currentIndex);
      currentIndex--;
  
      // And swap it with the current element.
      [array[currentIndex], array[randomIndex]] = [
        array[randomIndex], array[currentIndex]];
    }
}

// Source: https://stackoverflow.com/questions/1527803/generating-random-whole-numbers-in-javascript-in-a-specific-range
function getRandomArbitrary(min: number, max: number) {
    return Math.random() * (max - min) + min;
}

// Source: https://stackoverflow.com/questions/1527803/generating-random-whole-numbers-in-javascript-in-a-specific-range
function getRandomInt(min: number, max: number) {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

class NDVector {
    numberOfDimensions: number;
    normalVectors: tf.Tensor2D;
    dimensionIndices: tf.Tensor;
    planeIndices: tf.Tensor;

    constructor(numberOfDimensions: number) {
        this.numberOfDimensions = numberOfDimensions;
        this.normalVectors = tf.eye(numberOfDimensions, ...[, ,], 'float32');
        this.dimensionIndices = tf.stack(Array.from({ length: numberOfDimensions }, (_, i) => tf.tensor([i, i + numberOfDimensions], ...[,], 'float32')));
        this.planeIndices = this._initPlaneIndices();

    }

    _initPlaneIndices(): tf.Tensor {
        const _tensorList: tf.Tensor[] = getSubsetsOfSizeK<number>(Array.from(Array(this.numberOfDimensions).keys()), 2)
            .map((v: number[]) => tf.tensor(v, ...[,], 'int32'));
        return tf.stack(_tensorList);
    }
}


const geometry = new THREE.BoxGeometry();
const snakeMaterialHead = new THREE.MeshBasicMaterial({ color: 0x72e66c });
const snakeMaterialTail = new THREE.MeshBasicMaterial({ color: 0x8aace3 });
const snakeMaterialDead = new THREE.MeshBasicMaterial({ color: 0x1a1a1a });
const foodMaterial = new THREE.MeshBasicMaterial({ color: 0x403f23 });

class NDGameObject {
    numberOfDimensions: number;
    direction: NDVector;
    alive: boolean;
    fitness: number;
    foodEaten: number;
    history: tf.Tensor1D[];
    uuids: string[];

    constructor(numberOfDimensions: number) {
        this.numberOfDimensions = numberOfDimensions;
        
        this.direction = new NDVector(this.numberOfDimensions);
        this.alive = true;
        this.fitness = TTL;
        this.foodEaten = INITIAL_SCORE;
        this.history = this._resetHistory(tf.zeros([numberOfDimensions]));
        this.uuids = [];
    }

    _resetHistory(startPosition: tf.Tensor1D) {
        let tmp = []
        let pos = startPosition;
        for (let i = 0; i < this.foodEaten; i++) {
            tmp.push(pos);
            pos = pos.sub(tf.tensor([1, ...Array.from({length: this.numberOfDimensions-1}, (v) => 0)]))
        }
        return tmp
    }

    reset() {
        this.direction = new NDVector(this.numberOfDimensions);
        this.alive = true;
        this.fitness = TTL;
        this.foodEaten = INITIAL_SCORE;
        this.history = this._resetHistory(tf.zeros([this.numberOfDimensions]));
        this.uuids = [];
    }

    transformDirection(angles: tf.Tensor1D) {
        
        let _normalVectors = this.direction.normalVectors;
        const _basisVectors = _normalVectors.gather(this.direction.planeIndices)
        const _basisVectorsTransposed = _basisVectors.transpose([0, _basisVectors.rank - 1, _basisVectors.rank - 2]);

        const _m = _basisVectorsTransposed.shape[_basisVectors.rank - 1];
        const _n = _basisVectorsTransposed.shape[_basisVectors.rank - 2];

        const _vProjected = tf.matMul(_basisVectors, _normalVectors.expandDims(0).tile([_basisVectors.shape[0], 1, 1]));
        
        const _batchCosine: tf.Tensor1D = tf.cos(angles);
        const _batchSine: tf.Tensor1D = tf.sin(angles);
        
        
        const _rotationMatrixIdentity: tf.Tensor3D = tf.eye(_m).expandDims(0).tile([angles.shape[0], 1, 1]);
        const _range: tf.Tensor2D = tf.range(0, _rotationMatrixIdentity.shape[0], ...[,], "int32").expandDims(0).tile([4, 1]).transpose().reshape([1, _rotationMatrixIdentity.shape[0] * 4]).transpose()
        const _indices: tf.Tensor2D = tf.tensor2d([[0, 0], [0, 1], [1, 0], [1, 1]], ...[,], "int32").tile([_rotationMatrixIdentity.shape[0], 1]);
        const _repeatedIndices = tf.concat([_range, _indices], 1);
        
        
        const _rot: tf.Tensor1D = tf.stack<tf.Tensor1D>([_batchCosine, _batchSine.neg(), _batchSine, _batchCosine]).transpose().reshape([1, _rotationMatrixIdentity.shape[0] * 4]).squeeze();
        

        const _rotationMatrix = tf.tensorScatterUpdate(_rotationMatrixIdentity, _repeatedIndices, _rot);
        
        const _vRotatedProjected = tf.matMul(_rotationMatrix, _vProjected);
        const _vRotated = tf.matMul(_basisVectors, _vRotatedProjected, true, false);

        _normalVectors = normalize<tf.Tensor2D>(_vRotated.slice([0], [this.numberOfDimensions]).sum(0), 1);
        
        this.direction.normalVectors = _normalVectors;
        
    }

    update(foodPosition: tf.Tensor, currentPosition: tf.Tensor1D): tf.Tensor1D {
        
        // Snake game logic starts here
        const currentDirection = this.direction.normalVectors.slice([0], [1]).squeeze();
        const nextPosition = currentPosition.add<tf.Tensor1D>(currentDirection);

        // Enforce snake game rules
        const passiveDeath = this.fitness <=(this.foodEaten - INITIAL_SCORE) * TTL;
        const activeDeath = false;
        if (passiveDeath || activeDeath) {
            this.alive = false;
            return currentPosition
        }

        
        
        
        // Enforce snake fitness rules
        const nextFoodDistance = nextPosition.sub(foodPosition).square().sum(0).sqrt().arraySync() as number[];
        
        if (nextFoodDistance[0]) {
            this.foodEaten++;
            this.fitness = TTL + 2**(this.foodEaten - INITIAL_SCORE + 1) * TTL;
        } else {
            const currentFoodDistance = nextPosition.sub(foodPosition).square().sum(0).sqrt().arraySync() as number[];
            
            
            if (nextFoodDistance[0] < currentFoodDistance[0]) {
                this.fitness = this.fitness + 1;
            } else {
                this.fitness = this.fitness - 5;
            }
        }
        
        
        // Update position
        //this.position = nextPosition;

        // Keep history length fixed
        this.history.push(nextPosition);
        while (this.history.length > this.foodEaten) {
            this.history.shift()
        }
        
        
        return nextPosition
    }
}

class SnakeModel {
    id: number;
    numberOfDimensions: number;
    gameObject: NDGameObject;
    hiddenFeatures: number;
    inFeatures: number;
    outFeatures: number;
    feedForward: tf.Sequential;
    

    constructor(numberOfDimensions: number, hiddenFeatures: number, id: number) {
        this.id = id
        this.numberOfDimensions = numberOfDimensions;
        this.gameObject = new NDGameObject(numberOfDimensions);
        this.hiddenFeatures = hiddenFeatures;
        this.inFeatures = numberOfDimensions * 6 - 1;
        this.outFeatures = Math.floor((numberOfDimensions - 1) * numberOfDimensions / 2);
        this.feedForward = tf.sequential({
            layers: [
                tf.layers.dense({
                    units: this.hiddenFeatures,
                    inputShape: [this.inFeatures],
                    useBias: false,  // Matches PyTorch's bias=False
                    activation: 'relu' // ReLU activation
                }),
                tf.layers.dense({
                    units: this.outFeatures,
                    useBias: false,  // Matches PyTorch's bias=False
                    activation: 'relu' // ReLU activation
                })
            ]
        });
        
    }

    forward(foodPosition: tf.Tensor, currentPosition: tf.Tensor1D): tf.Tensor1D {
        const inputs = this._gameLayers(currentPosition, foodPosition, this.gameObject.direction.normalVectors, tf.stack(this.gameObject.history)) as tf.Tensor;
        const logits = (this.feedForward.predict(inputs) as tf.Tensor2D).squeeze();
        const angles = normalize<tf.Tensor1D>(logits, 0).sub(0.5).mul<tf.Tensor1D>(Math.PI);
        this.gameObject.transformDirection(angles);
        return this.gameObject.update(foodPosition, currentPosition);
    }

    _gameLayers(currentPosition: tf.Tensor1D, foodPosition: tf.Tensor, direction: tf.Tensor, snakeHistory: tf.Tensor): tf.Tensor {
        
        // Extend basis vectors to negative axis
        const signedVelocity = tf.concat([direction, direction.mul(-1)]);

        // Extend unit vectors to negative axis
        const signedUnitVectors = tf.concat([tf.eye(this.numberOfDimensions), tf.eye(this.numberOfDimensions).mul(-1)]);

        // Food data processing and vectorizing (n_dims * 2)
        const foodNormals = cosineSimilarity(currentPosition.sub(foodPosition).expandDims(0).tile([this.numberOfDimensions * 2, 1]), signedVelocity, 1);
        const nearbyFood = foodNormals.clipByValue(-1, 1);
        const foodDistance = currentPosition.sub(foodPosition).pow(2).sum(-1).sqrt();

        // Body data processing and vectorizing (5) (n_dims * 2 - 1)
        const bodyNormals = signedVelocity.add(currentPosition.expandDims(0).tile([this.numberOfDimensions * 2, 1]));
        const snakeHistoryHeadOmitted = snakeHistory.slice([0, 0], [snakeHistory.shape[0] - 1, this.numberOfDimensions]); // Omit the head of the snake
        const bodyDistances = cdist(bodyNormals, snakeHistoryHeadOmitted);
        const nearbyBody = tf.any(tf.concat([bodyDistances.slice([0], [this.numberOfDimensions]), bodyDistances.slice([this.numberOfDimensions + 1], [this.numberOfDimensions - 1])]).equal(0), 1).cast('float32');

        // Bounds data processing and vectorizing (3) (n_dims * 2 - 1)
        // TODO: Add new raycasting function to determine bounds
        const nearbyBounds = this._calculateBounds(BOUNDS_SCALE, currentPosition).softmax(); // This is a placeholder for new logic

        const inputs = tf.concat([nearbyFood, nearbyBounds, nearbyBody]);
        
        return inputs.expandDims(0)
    }

    _calculateBounds(boundTensor: tf.TensorLike, currentPosition: tf.Tensor1D): tf.Tensor1D {
        const identity = tf.eye(this.numberOfDimensions);
        const extendedIdentity = tf.concat([identity, identity.neg()]).mul<tf.Tensor2D>(boundTensor);
        const velocity = this.gameObject.direction.normalVectors;
        const extendedVelocity = tf.concat([velocity, velocity.neg()])
        return distanceFromPlane(extendedIdentity, extendedIdentity.neg(), currentPosition, extendedVelocity)
    }

    
}



class Trainer {
    numberOfDimensions: number;
    populationSize: number;
    crossoverProbability: number;
    differentialWeight: number;
    foodArray: tf.Tensor1D[];
    currentCandidates: SnakeModel[];
    nextCandidates: SnakeModel[];
    currentCandidatesPositions: tf.Tensor2D;
    nextCandidatesPositions: tf.Tensor2D;
    currentCandidateFitnesses: tf.Tensor1D;
    nextCandidateFitnesses: tf.Tensor1D;
    foodIds: tf.Tensor1D;
    stats: {[s: string]: number};
    status: string;

    constructor(numberOfDimensions: number, populationSize: number, crossoverProbability: number, differentialWeight: number) {
        this.numberOfDimensions = numberOfDimensions;
        this.populationSize = populationSize;
        this.crossoverProbability = crossoverProbability;
        this.differentialWeight = differentialWeight;
        this.foodArray = [tf.randomUniform([numberOfDimensions])];
        this.currentCandidates = this._generateCandidates();
        this.nextCandidates = this._generateCandidates(populationSize);
        this.currentCandidatesPositions = tf.zeros([populationSize, numberOfDimensions]);
        this.nextCandidatesPositions = tf.zeros([populationSize, numberOfDimensions]);
        this.currentCandidateFitnesses = tf.ones([populationSize]).mul(TTL);
        this.nextCandidateFitnesses = tf.ones([populationSize]).mul(TTL);
        this.foodIds = tf.zeros([populationSize], "int32");
        this.stats = {}
        this.status = 'ready';
    }

    // Helper function generates next candidate from candidate at specified index to participate in the next round of training
    generateNextCandidate(currentCandidateIndex: number) {
        const indices = Array.from(Array(this.populationSize).keys()).filter((v: number) => v != currentCandidateIndex);
        shuffle(indices);
        const agentA = this.currentCandidates[indices[0]].feedForward.getWeights();
        const agentB = this.currentCandidates[indices[1]].feedForward.getWeights();
        const agentC = this.currentCandidates[indices[2]].feedForward.getWeights();
        let weights: tf.Tensor[] = [];
        this.currentCandidates[currentCandidateIndex].feedForward.getWeights().map((layer: tf.Tensor, layerIndex: number) => {
            const rValues = tf.randomUniform(layer.shape, ...[, ,], "float32");
            const mask = rValues.greaterEqual(this.crossoverProbability).cast('int32');
            const maskNegation = layer.mul(rValues.less(this.crossoverProbability).cast('float32'));
            weights.push(agentB[layerIndex].sub(agentC[layerIndex]).mul(this.differentialWeight).add(agentA[layerIndex]).mul(mask).add(maskNegation));
        })
        this.nextCandidates[currentCandidateIndex].feedForward.setWeights(weights);
    }

    // Reset the indexed gameobject
    resetCandidateGameObject(candidateIndex: number, renderer?: Renderer) {
        if (candidateIndex >= this.populationSize) {
            throw new RangeError("Specified index exceeds the population size");
        }
        
        this.currentCandidates[candidateIndex].gameObject.reset();
        this.nextCandidates[candidateIndex].gameObject.reset();

        if (!renderer) {
            return
        }

        renderer.addGameObject(this.currentCandidates[candidateIndex].gameObject);
        renderer.addGameObject(this.nextCandidates[candidateIndex].gameObject);
    }

    // Generate all of the next candidates to participate in the next round of training
    setupCandidates(renderer?: Renderer) {
        renderer?.resetScene();
        for (let currentCandidateIndex = 0; currentCandidateIndex < this.populationSize; currentCandidateIndex++) {
            this.generateNextCandidate(currentCandidateIndex);
            this.resetCandidateGameObject(currentCandidateIndex, renderer);
        }
        console.log("candidates setup");
        this.status = 'active';
    }

    // Helper function that updates the simulation for the indexed candidate by one frame
    evaluateCandidateAnimationFrame(candidate: SnakeModel, previousPosition: tf.Tensor1D, renderer?: Renderer): tf.Tensor1D {
        console.time("evaluateCandidateAnimationFrame");

        // Resize the food array if needed
        if (candidate.gameObject.foodEaten - INITIAL_SCORE >= this.foodArray.length) {
            this.foodArray.push(tf.randomUniform([this.numberOfDimensions]))
        }

        // Store the current target food
        const candidateTargetFood = this.foodArray[candidate.gameObject.foodEaten - INITIAL_SCORE];
        
        
        if (!candidate.gameObject.alive) {
            return previousPosition
        }

        // Perform forward pass on the current candidate model
        const nextPosition = candidate.forward(candidateTargetFood, previousPosition);
        this.status = 'active';

        if (!renderer) {
            return nextPosition
        }

        // Sync the game changes with the renderer
        renderer.updateGameObject(candidate.gameObject);
        console.timeEnd("evaluateCandidateAnimationFrame");
        return nextPosition
    }

    
    // Update the simulation for all candidates by one frame
    evaluateCandidatesAnimationFrame(candidates: SnakeModel[], positions: tf.Tensor2D, fitnesses: tf.Tensor1D, renderer?: Renderer): tf.Tensor1D {
        
        let p: tf.Tensor1D[] = [];

        // Assume the training evaluation ends following this step, unless each candidate proves otherwise
        this.status = 'inactive';
        for (let candidateIndex = 0; candidateIndex < this.populationSize; candidateIndex++) {
            const previousPosition = positions.slice([candidateIndex, 0], [1, this.numberOfDimensions]).squeeze() as tf.Tensor1D;
            p.push(this.evaluateCandidateAnimationFrame(candidates[candidateIndex], previousPosition, renderer));
        }
        
        
        const nextPositions = tf.stack(p) // (population_num, dimension_num)
        const foodPositions = tf.gather(tf.stack(this.foodArray), this.foodIds) // (population_num, dimension_num)
        const nextDistances = tf.squaredDifference(nextPositions, foodPositions).sum(1).sqrt(); // Euclidean distance
        const previousDistances = tf.squaredDifference(positions, foodPositions).sum(1).sqrt();
        const distanceDifferences = tf.sub(previousDistances, nextDistances);
        const improvingMask = distanceDifferences.greater(0).mul(1);
        const degradingMask = distanceDifferences.lessEqual(0).mul(-5);
        
        return fitnesses.add(improvingMask.add(degradingMask));
    }

    // Evaluate the current and next candidates
    evaluatePairsAnimationFrame(renderer?: Renderer) {
        this.currentCandidateFitnesses = this.evaluateCandidatesAnimationFrame(this.currentCandidates, this.currentCandidatesPositions, this.currentCandidateFitnesses, renderer);
        this.nextCandidateFitnesses = this.evaluateCandidatesAnimationFrame(this.nextCandidates, this.nextCandidatesPositions, this.nextCandidateFitnesses, renderer);
    }

    evaluateCandidateAnimationEnd(candidateIndex: number) {
        const currentCandidateFitness = this.currentCandidates[candidateIndex].gameObject.fitness;
        const nextCandidateFitness = this.nextCandidates[candidateIndex].gameObject.fitness;
        if (nextCandidateFitness >= currentCandidateFitness) {
            this.currentCandidates[candidateIndex].feedForward.setWeights(this.nextCandidates[candidateIndex].feedForward.getWeights());
        }
    }

    evaluateCandidatesAnimationEnd() {
        // Ensure this method only occurs once per training simulation
        if (this.status != 'inactive') {
            return
        }

        const fitnessDifference = this.currentCandidateFitnesses.less(this.nextCandidateFitnesses).cast("float32");
        fitnessDifference.print();

        for (let candidateIndex = 0; candidateIndex < this.populationSize; candidateIndex++) {
            this.evaluateCandidateAnimationEnd(candidateIndex);
        }
        this.status = 'ready'; // Flag as ready to start next training simulation
    }

    _generateCandidates(offset: number = 0) {
        let pop = [];
        for (let i = 0; i < this.populationSize; i++) {
            pop.push(new SnakeModel(this.numberOfDimensions, HIDDEN_LAYER_SIZE, i + offset));
        }
        return pop
    }

    debug() {
        console.log("test");
    }

    getSessionStats(data: number[][]) {
        const fitnesses = tf.tensor2d(data);
        const currentFitnesses = fitnesses.slice([0, 0], [fitnesses.shape[0], 1]).squeeze<tf.Tensor1D>().arraySync();
        const nextFitnesses = fitnesses.slice([0, 1], [fitnesses.shape[0], 1]).squeeze<tf.Tensor1D>().arraySync();
        const currentAverageFitness = currentFitnesses.reduce((partialSum: number, a: number) => partialSum + a, 0) / this.populationSize;
        const nextAverageFitness = nextFitnesses.reduce((partialSum: number, a: number) => partialSum + a, 0) / this.populationSize;
        const diff = nextAverageFitness - currentAverageFitness;
        return { currentAverageFitness, nextAverageFitness, diff }
    }
}

// Return the cross-product of two 3-dimensional vectors
function crossProduct3D(P: tf.Tensor1D, Q: tf.Tensor1D): tf.Tensor1D {
    const p = P.arraySync();
    const q = Q.arraySync();
    
    const crossI = p[1] * q[2] - p[2] * q[1];
    const crossJ = p[2] * q[0] - p[0] * q[2];
    const crossK = p[0] * q[1] - p[1] * q[0];

    return tf.tensor1d([crossI, crossJ, crossK]);
}

// For vectors of higher dimensions (>3), all but three entries should be shared among the three input vectors
// That way it can be safely ignored in calculating the cross-product
function getNormalVector(P: tf.Tensor1D, Q: tf.Tensor1D, R: tf.Tensor1D, isUnitLength: boolean = true): tf.Tensor1D {
    const PQ = tf.sub<tf.Tensor1D>(Q, P);
    const PR = tf.sub<tf.Tensor1D>(R, P);

    if (isUnitLength) {
        // Reduce PQ and PR to unit length 1
        const unitVectorPQ = tf.div<tf.Tensor1D>(PQ, PQ.norm('euclidean'));
        const unitVectorPR = tf.div<tf.Tensor1D>(PR, PR.norm('euclidean'));
        const crossProduct = crossProduct3D(unitVectorPQ, unitVectorPR); // Calculate the cross-product
        return crossProduct
    }

    const crossProduct = crossProduct3D(PQ, PR); // Calculate the cross-product

    return crossProduct;
}

function projectOntoPlane(P0: tf.Tensor2D, L0: tf.Tensor2D, n: tf.Tensor2D) {
    const L = tf.div<tf.Tensor2D>(L0, L0.norm('euclidean', 1).expandDims(1));
    const diff1 = tf.sub<tf.Tensor1D>(P0, L0);
    const dot1 = tf.mul(diff1, n).sum(1); // Batched row-pairwise dot product
    const dot2 = tf.mul(L, n).sum(1);
    const div1 = tf.div<tf.Tensor1D>(dot1, dot2);
    const P = tf.add(L0, tf.mul(L, div1.expandDims(1)));
    return P
}

function distanceFromPlane(planePoint: tf.Tensor2D, normalPoint: tf.Tensor2D, position: tf.Tensor1D, velocity: tf.Tensor2D): tf.Tensor1D {
    // Offset the plane to ensure accurate distance
    const offsetPoint = tf.sub<tf.Tensor2D>(planePoint, position.expandDims(-2).tile([planePoint.shape[0], 1]));
    // Find the point that the line from the velocity vector intersects the plane
    const intersectionPoint = projectOntoPlane(offsetPoint, velocity, normalPoint);
    // Calculate the distance between the point on the plane and the velocity vector
    const distance = tf.sub<tf.Tensor2D>(intersectionPoint, velocity).square().sum<tf.Tensor1D>(1).sqrt();
    return distance
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

function main() {
    const trainer = new Trainer(4, 100, 0.9, 0.8);

    const animate = () => {
        
        

        
        //console.time("evaluateCandidatesAnimationFrame")
        trainer.evaluatePairsAnimationFrame();
        //console.timeEnd("evaluateCandidatesAnimationFrame")
        
        if (trainer.status == 'active') {
            animate()
        }
    }

    animate();
    trainer.evaluateCandidatesAnimationEnd()

    /*
    // Unit test for calculating normal vector of a plane
    const p = tf.tensor1d([1, 2, 3]);
    const q = tf.tensor1d([2, 1, 1]);
    const r = tf.tensor1d([0, 3, 2]);
    const normal = getNormalVector(p, q, r);
    normal.print();
    */

    
}

export { SnakeModel, Trainer, NDGameObject, NDVector, BOUNDS_SCALE };