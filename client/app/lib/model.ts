import * as tf from '@tensorflow/tfjs';

const TTL = 100;
const INITIAL_SCORE = 5;

const INPUT_LAYER_SIZE = 10;  // Example size
const HIDDEN_LAYER_SIZE = 32;
const OUTPUT_LAYER_SIZE = 5;

const BOUNDS_SCALE = 20;

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

function normalize(A: tf.Tensor, dim: number) {
    const norm = tf.norm(A, 'euclidean', 1);
    const normalized = A.div(norm);
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
    normalVectors: tf.Tensor;
    dimensionIndices: tf.Tensor;
    planeIndices: tf.Tensor;

    constructor(numberOfDimensions: number) {
        this.numberOfDimensions = numberOfDimensions;
        this.normalVectors = tf.eye(numberOfDimensions, ...[, ,], 'float32');
        this.dimensionIndices = tf.stack(Array.from({ length: numberOfDimensions }, (_, i) => tf.tensor([i, i + numberOfDimensions], ...[,], 'float32')));
        this.planeIndices = this._initPlaneIndices();
    }

    _initPlaneIndices(): tf.Tensor {
        let _planeIndices = new Set<Set<number>>([]);
        for (let i = 0; i < this.numberOfDimensions; i++) {
            for (let j = 0; j < this.numberOfDimensions; j++) {
                if (i == j) continue;
                _planeIndices.add(new Set([i, j]))
            }
        }

        const _tensorList: tf.Tensor[] = Array.from(_planeIndices.values())
            .map((v: Set<number>) => tf.tensor(Array.from(v), ...[,], 'int32'));
        return tf.stack(_tensorList);
    }
}

class NDGameObject {
    numberOfDimensions: number;
    position: tf.Tensor;
    direction: NDVector;

    constructor(numberOfDimensions: number) {
        this.numberOfDimensions = numberOfDimensions;
        this.position = tf.zeros([numberOfDimensions]);
        this.direction = new NDVector(numberOfDimensions);
    }

    transformDirection(angles: tf.Tensor) {
        let _normalVectors = this.direction.normalVectors;
        const _basisVectors = _normalVectors.gather(this.direction.planeIndices)
        const _basisVectorsTransposed = _basisVectors.transpose([0, _basisVectors.rank - 1, _basisVectors.rank - 2]);

        const _m = _basisVectorsTransposed.shape[_basisVectors.rank - 1];
        const _n = _basisVectorsTransposed.shape[_basisVectors.rank - 2];

        const _vProjected = tf.matMul(_basisVectors, _normalVectors);

        const _batchCosine = tf.cos(angles);
        const _batchSine = tf.sin(angles);
        const _rotationMatrixBuffer = tf.eye(_m).expandDims(0).tile([angles.shape[0], 1, 1]).bufferSync();

        // Might be the root of all problems for performance but we'll address this later
        const _rot: tf.Tensor[] = [_batchCosine, _batchSine.neg(), _batchSine, _batchCosine];
        for (let i = 0; i < 4; i++) {
            const firstBit = (i >> 1) & 1;
            const secondBit = i & 1;
            for (let j = 0; j < angles.shape[0]; j++) {
                const _val = _rot[i].slice([j], [1]).dataSync()[0];
                _rotationMatrixBuffer.set(_val, ...[j, firstBit, secondBit])
            }
        }

        const _rotationMatrix = _rotationMatrixBuffer.toTensor();
        const _vRotatedProjected = tf.matMul(_rotationMatrix, _vProjected);
        const _vRotated = tf.matMul(_basisVectors, _vRotatedProjected, true, false);

        /*_normalVectors = _normalVectors
            .expandDims(0)
            .tile([this.direction.planeIndices.shape[0], 1, 1])
            .add(_vRotated)
            .softmax();*/

        _normalVectors = normalize(_vRotated.slice([0], [angles.shape[1]]).sum(0), 2);

        this.direction.normalVectors = _normalVectors;
    }

    transformPosition(distance: number = 1) {
        this.position = this.position.add(this.direction.normalVectors.slice([0, 0], [1, this.numberOfDimensions])).squeeze();
    }
}

class SnakeModel {
    numberOfDimensions: number;
    gameObject: NDGameObject;
    hiddenFeatures: number;
    inFeatures: number;
    outFeatures: number;
    feedForward: tf.Sequential;
    alive: boolean;
    fitness: number;
    foodEaten: number;
    history: tf.Tensor[];

    constructor(numberOfDimensions: number, hiddenFeatures: number) {
        this.numberOfDimensions = numberOfDimensions;
        this.gameObject = new NDGameObject(numberOfDimensions);
        this.hiddenFeatures = hiddenFeatures;
        this.inFeatures = numberOfDimensions * 6 - 2;
        this.outFeatures = Math.floor((numberOfDimensions - 1) * numberOfDimensions / 2);
        this.feedForward = tf.sequential({
            layers: [
                tf.layers.dense({
                    units: this.hiddenFeatures,
                    inputShape: [16],
                    useBias: false,  // Matches PyTorch's bias=False
                    activation: 'relu' // ReLU activation
                }),
                tf.layers.dense({
                    units: this.outFeatures,
                    useBias: false,  // Matches PyTorch's bias=False
                    activation: 'relu' // ReLU activation
                }),
                tf.layers.softmax({
                    dtype: 'float32'
                })
            ]
        });
        this.alive = true;
        this.fitness = TTL;
        this.foodEaten = INITIAL_SCORE;
        this.history = [];

        let pos = this.gameObject.position;
        for (let i = 0; i < this.foodEaten; i++) {
            this.history.push(pos);
            pos = pos.sub(tf.tensor([1, ...Array.from({length: this.numberOfDimensions-1}, (v) => 0)]))
        }
    }

    forward(foodPosition: tf.Tensor) {
        const fakeHistory = this.gameObject.position.expandDims(0).tile([10, 1]).mul(tf.range(0, 10).expandDims(0).transpose().tile([1, 3]));
        const inputs = this._gameLayers(this.gameObject.position, foodPosition, this.gameObject.direction.normalVectors, fakeHistory) as tf.Tensor;
        const logits = this.feedForward.predict(inputs) as tf.Tensor;
        const angles = logits.sub(0.5).mul(Math.PI);
        this.gameObject.transformDirection(angles);
        this.gameObject.transformPosition();

        // Snake game logic starts here
        const currentDirection = this.gameObject.direction.normalVectors.slice([0], [1]).squeeze();
        const nextPosition = this.gameObject.position.add(currentDirection);

        // Enforce snake game rules
        const passiveDeath = this.fitness <=(this.foodEaten - INITIAL_SCORE) * TTL;
        const activeDeath = false;
        if (passiveDeath || activeDeath) {
            this.alive = false;
            return
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
        this.gameObject.position = nextPosition;

        // Keep history length fixed
        while (this.history.length > this.foodEaten) {
            this.history.shift()
        }
    }

    _gameLayers(snakePosition: tf.Tensor, foodPosition: tf.Tensor, direction: tf.Tensor, snakeHistory: tf.Tensor): tf.Tensor {
        // Extend basis vectors to negative axis
        const signedVelocity = tf.concat([direction, direction.mul(-1)]);

        // Extend unit vectors to negative axis
        const signedUnitVectors = tf.concat([tf.eye(this.numberOfDimensions), tf.eye(this.numberOfDimensions).mul(-1)]);

        // Food data processing and vectorizing (n_dims * 2)
        
        const foodNormals = cosineSimilarity(snakePosition.sub(foodPosition).expandDims(0).tile([this.numberOfDimensions * 2, 1]), signedVelocity, 1);
        const nearbyFood = foodNormals.clipByValue(-1, 1);
        const foodDistance = snakePosition.sub(foodPosition).pow(2).sum(-1).sqrt();

        // Body data processing and vectorizing (5) (n_dims * 2 - 1)
        const bodyNormals = signedVelocity.add(snakePosition.expandDims(0).tile([this.numberOfDimensions * 2, 1]));
        const snakeHistoryHeadOmitted = snakeHistory.slice([0, 0], [snakeHistory.shape[0] - 1, this.numberOfDimensions]); // Omit the head of the snake
        const bodyDistances = cdist(bodyNormals, snakeHistoryHeadOmitted);
        const nearbyBody = tf.any(tf.concat([bodyDistances.slice([0], [this.numberOfDimensions]), bodyDistances.slice([this.numberOfDimensions + 1], [this.numberOfDimensions - 1])]).equal(0), 1).cast('float32');

        // Bounds data processing and vectorizing (3) (n_dims * 2 - 1)
        // TODO: Add new raycasting function to determine bounds
        const nearbyBounds = tf.zeros([this.numberOfDimensions*2-1]); // This is a placeholder for new logic

        const inputs = tf.concat([nearbyFood, nearbyBounds, nearbyBody]);
        return inputs.expandDims(0)
    }

    _calculateBoundsHelper(bounds: tf.Tensor, signedUnitVectors: tf.Tensor) {
        const boundsDistance = (bounds.mul(signedUnitVectors)).sub(this.gameObject.position.mul(tf.abs(signedUnitVectors)));
        const boundsPositive = tf.sum(boundsDistance)
    }
}

class Trainer {
    numberOfDimensions: number;
    populationSize: number;
    crossoverProbability: number;
    differentialWeight: number;
    foodArray: tf.Tensor[];
    currentCandidates: SnakeModel[];
    nextCandidates: SnakeModel[];

    constructor(numberOfDimensions: number, populationSize: number, crossoverProbability: number, differentialWeight: number) {
        this.numberOfDimensions = numberOfDimensions;
        this.populationSize = populationSize;
        this.crossoverProbability = crossoverProbability;
        this.differentialWeight = differentialWeight;
        this.foodArray = [tf.randomUniform([numberOfDimensions])];
        this.currentCandidates = this._generateCandidates();
        this.nextCandidates = this._generateCandidates();
    }

    async start(numberOfIterations: number) {
        for (let i = 0; i < numberOfIterations; i++) {
            await this.step()
        }
    }

    async step() {
        await Promise.all(this.currentCandidates.map((agentX, index: number) => {
            const indices = Array.from(Array(this.populationSize).keys()).filter((v: number) => v != index);
            shuffle(indices);
            const agentA = this.currentCandidates[indices[0]].feedForward.getWeights();
            const agentB = this.currentCandidates[indices[1]].feedForward.getWeights();
            const agentC = this.currentCandidates[indices[2]].feedForward.getWeights();
            let weights: tf.Tensor[] = [];
            this.currentCandidates[index].feedForward.getWeights().map((layer: tf.Tensor, layerIndex: number) => {
                const rValues = tf.randomUniform([HIDDEN_LAYER_SIZE], ...[, ,], "float32");
                const mask = rValues.greaterEqual(this.crossoverProbability).cast('int32')
                const maskNegation = rValues.less(this.crossoverProbability).cast('int32').mul(layer);
                weights.push(agentB[layerIndex].sub(agentC[layerIndex]).mul(this.differentialWeight).add(agentA[layerIndex]).mul(mask).add(maskNegation));
            })
            this.nextCandidates[index].feedForward.setWeights(weights);
            return this.session(index)
        }))
    }

    async session(id: number) {
        while (this.currentCandidates[id].alive || this.nextCandidates[id].alive) {
            // Resize the food array if needed
            if (this.currentCandidates[id].foodEaten >= this.foodArray.length || this.nextCandidates[id].foodEaten >= this.foodArray.length) {
                this.foodArray.push(tf.randomUniform([this.numberOfDimensions]))
            }

            const currentCandidateTargetFood = this.foodArray[this.currentCandidates[id].foodEaten];
            const nextCandidateTargetFood = this.foodArray[this.nextCandidates[id].foodEaten];
            this.currentCandidates[id].forward(currentCandidateTargetFood);
            this.nextCandidates[id].forward(nextCandidateTargetFood);
        }

        if (this.nextCandidates[id].fitness > this.currentCandidates[id].fitness) {
            this.currentCandidates[id].feedForward.setWeights(this.nextCandidates[id].feedForward.getWeights());
        }
    }

    _generateCandidates() {
        let pop = [];
        for (let i = 0; i < this.populationSize; i++) {
            pop.push(new SnakeModel(this.numberOfDimensions, HIDDEN_LAYER_SIZE));
        }
        return pop
    }
}


function main() {
    const trainer = new Trainer(3, 10, 0.9, 0.8);
    trainer.start(10);
}

main();

export { SnakeModel, Trainer, NDGameObject, NDVector };