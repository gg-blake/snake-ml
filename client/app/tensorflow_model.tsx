import React, { useEffect, useState, useRef } from 'react';

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { resolve } from 'path';
import RenderResult from 'next/dist/server/render-result';
import * as tf from '@tensorflow/tfjs';
import {loadGraphModel} from '@tensorflow/tfjs-converter';
import * as ort from 'onnxruntime-web/training';
import { ToonShaderHatching } from 'three/examples/jsm/Addons.js';

const TTL = 100;
const INITIAL_SCORE = 5;

const INPUT_LAYER_SIZE = 10;  // Example size
const HIDDEN_LAYER_SIZE = 32;
const OUTPUT_LAYER_SIZE = 5;

const BOUNDS_SCALE = 20;

class NDVector {
    numberOfDimensions: number;
    normalVectors: tf.Tensor;
    dimensionIndices: tf.Tensor;
    planeIndices: tf.Tensor;

    constructor(numberOfDimensions: number) {
        this.numberOfDimensions = numberOfDimensions;
        this.normalVectors = tf.eye(numberOfDimensions, ...[,,], 'float32');
        this.dimensionIndices = tf.stack(Array.from({ length: numberOfDimensions }, (_, i) => tf.tensor([i, i+numberOfDimensions], ...[,], 'float32')));
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
            .map((v: Set<number>) => tf.tensor(Array.from(v), ...[,], 'float32'));
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
        const _basisVectorsTransposed = _basisVectors.transpose([_basisVectors.rank-1, _basisVectors.rank-2]);

        const _m = _basisVectors.shape[_basisVectors.rank-1];
        const _n = _basisVectors.shape[_basisVectors.rank-2];

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
        const _vRotated = tf.matMul(_basisVectorsTransposed, _vRotatedProjected);

        _normalVectors = _normalVectors
            .expandDims(0)
            .tile([this.direction.planeIndices.shape[0], 1, 1])
            .add(_vRotated)
            .euclideanNorm(_vRotated.rank-1);

        this.direction.normalVectors = _normalVectors;
    }
}

class DEModel {
    numberOfDimensions: number;
    populationSize: number;
    crossoverProbability: number;
    differentialWeight: number;

    constructor(numberOfDimensions: number, populationSize: number, crossoverProbability: number, differentialWeight: number) {
        this.numberOfDimensions = numberOfDimensions;
        this.populationSize = populationSize;
        this.crossoverProbability = crossoverProbability;
        this.differentialWeight = differentialWeight;
    }

    train(numberOfIterations: number) {
        let x = tf.randomUniform([this.populationSize, HIDDEN_LAYER_SIZE]);
        for (let i = 0; i < numberOfIterations; i++) {
            for (let currentCandidateIndex = 0; currentCandidateIndex < this.populationSize; currentCandidateIndex++) {
                const leftWeights = x.slice([0, 0], [currentCandidateIndex, HIDDEN_LAYER_SIZE]);
                const rightWeights = x.slice([currentCandidateIndex+1, 0], [this.populationSize, HIDDEN_LAYER_SIZE]);
                const weightsMask = tf.concat([leftWeights, rightWeights]);

                let indices = Array.from({ length: this.populationSize-1 }, (i: number) => i);
                tf.util.shuffle(indices);

                const p = weightsMask.gather(tf.tensor(indices.slice(0, 3)));
                
                const currentCandidate = x.slice([currentCandidateIndex, 0], [1, HIDDEN_LAYER_SIZE])
                const p1 = p.slice([0, 0], [1, HIDDEN_LAYER_SIZE]);
                const p2 = p.slice([1, 0], [1, HIDDEN_LAYER_SIZE]);
                const p3 = p.slice([2, 0], [1, HIDDEN_LAYER_SIZE]);

                const rValues = tf.randomUniform([HIDDEN_LAYER_SIZE], ...[,,], "float32");
                const mask = rValues.greaterEqual(this.crossoverProbability)
                    .cast('int32')

                const maskNegation = rValues.less(this.crossoverProbability)
                    .cast('int32')
                    .mul(currentCandidate);
                
                const newCandidate = p2
                    .sub(p3)
                    .mul(this.differentialWeight)
                    .add(p1)
                    .mul(mask)
                    .add(maskNegation);

                const keep = compareCandidates(this.numberOfDimensions, currentCandidate, newCandidate) == false;
            }
        }
    }
}

function compareCandidatesBuildModel(weights: tf.Tensor): tf.Sequential {
    const feedForward = tf.sequential();
        
    // First Linear (Dense) Layer
    feedForward.add(tf.layers.dense({
        units: HIDDEN_LAYER_SIZE,
        inputShape: [INPUT_LAYER_SIZE],
        useBias: false,  // Matches PyTorch's bias=False
        activation: 'relu' // ReLU activation
    }));
    
    // Second Linear (Dense) Layer
    feedForward.add(tf.layers.dense({
        units: OUTPUT_LAYER_SIZE,
        useBias: false,  // Matches PyTorch's bias=False
        activation: 'relu' // ReLU activation
    }));

    return feedForward
}

function compareCandidatesModelForward(gameObject: NDGameObject, nearbyBody: tf.Tensor, foodPosition: tf.Tensor, bounds: tf.Tensor) {
    // Extend basis vectors to negative axis
    const signedVelocity = tf.concat([gameObject.direction.normalVectors, gameObject.direction.normalVectors.mul(-1)]);
    
    // Extend unit vectors to negative axis
    const signedUnitVectors = tf.concat([tf.eye(gameObject.numberOfDimensions), tf.eye(gameObject.numberOfDimensions).mul(-1)]);

    
}

function compareCandidates(numberOfDimensions: number, currentCandidate: tf.Tensor, newCandidate: tf.Tensor): boolean {
    // Define model parameters
    const INPUT_LAYER_SIZE = 10;  // Example size
    const HIDDEN_LAYER_SIZE = 32;
    const OUTPUT_LAYER_SIZE = 5;

    const currentCandidateGameObject = new NDGameObject(numberOfDimensions);
    const newCandidateGameObject = new NDGameObject(numberOfDimensions);

    let currentScore = 0;
    let foodPositions = [tf.randomUniformInt([numberOfDimensions], -BOUNDS_SCALE, BOUNDS_SCALE)];
    
    while (true) {
        
    }
    
    return false
}

function createIdentityMatrix(n: number): Float32Array {
    const identityMatrix = new Float32Array(n * n);
    for (let i = 0; i < n; i++) {
        identityMatrix[i * n + i] = 1; // Set the diagonal elements to 1
    }
    return identityMatrix;
}

function loadMesh(numberOfDimensions: number, mesh: THREE.Mesh<THREE.BoxGeometry, THREE.MeshBasicMaterial, THREE.Object3DEventMap>, position: number[]) {
    console.assert(numberOfDimensions == position.length, "Position vector should match the specified number of dimensions");
    
    // Initialize snake head mesh positions
    if (numberOfDimensions == 2) {
        mesh.position.set(position[0], position[1], 0)
    } else if (numberOfDimensions > 2) {
        mesh.position.set(position[0], position[1], position[2])
    } else {
        console.error("Dimensions must be equal or greater than 2")
    }
}

const foodPosition = tf.tensor([5, 4, 3], [3], 'float32');

class Snake {
    snakeScore: number;
    numberOfDimensions: number;
    snakePosition: tf.Tensor<tf.Rank>;
    snakeHistory: tf.Tensor<tf.Rank>[];
    snakeBounds: tf.Tensor<tf.Rank>;
    snakeVelocity: tf.Tensor<tf.Rank>;
    snakeFitness: number;
    snakeDistanceFromFood: number | null;
    snakeIsAlive: boolean
    session: ort.InferenceSession | undefined;

    constructor(numberOfDimensions: number, boundsSquareSize: number) {
        this.snakeScore = INITIAL_SCORE;
        this.numberOfDimensions = numberOfDimensions;
        this.snakePosition = tf.zeros([numberOfDimensions], 'float32');  // Shape (1, 4)
        this.snakeHistory = [this.snakePosition];
        this.snakeBounds = tf.tensor([...Array.from({ length : this.numberOfDimensions }, () => boundsSquareSize)], [numberOfDimensions], 'int32');  // Shape (1, 4)
        this.snakeVelocity = tf.eye(numberOfDimensions);
        this.snakeFitness = TTL;
        this.snakeDistanceFromFood = null;
        this.snakeIsAlive = true;
    }

    async update() {
        const MODEL_URL = '/web_model/model.json';
        const model = await loadGraphModel(MODEL_URL);
        console.log(model);
        for (var i = 0; i < 20; i++) {
            let logits = model.predict({
                "snake_pos": this.snakePosition, 
                "nearby_body": tf.tensor([0, 0, 0, 0, 0]), 
                "food_pos": foodPosition, 
                "bounds": this.snakeBounds, 
                "vel": this.snakeVelocity
            }) as tf.Tensor<tf.Rank>[];
            this.snakePosition = logits[0].slice([0], [this.numberOfDimensions]);
            this.snakeVelocity = logits[1];
            this.snakePosition.print();
        }
    }
    
    gameUpdate(output: ort.InferenceSession.OnnxValueMapType) {
        this.snakePosition = output.next_position.cpuData;
        this.snakeVelocity = output.velocity.cpuData;

        let currentDistanceFromFood: number = output.food_dist.cpuData[0];
        let currentBoundsDistance: Float32Array = output.bounds_dist.cpuData
        if (!this.snakeDistanceFromFood) {
            this.snakeDistanceFromFood = currentDistanceFromFood;
        }

        if (this.snakeDistanceFromFood - currentDistanceFromFood > 0) {
            this.snakeFitness++;
        } else {
            this.snakeFitness = this.snakeFitness - 10;
        }

        console.log(currentBoundsDistance)
        

        
        //console.log(`Current Bound: (${currentBoundsDistance[0]}, ${currentBoundsDistance[1]}, ${currentBoundsDistance[2]})\nCurrent Position: (${this.snakePosition[0]}, ${this.snakePosition[1]}, ${this.snakePosition[2]})`);
    }
}



const snakePopulation: Snake[] = [];
let isLoaded = false;
const foodPositions = [new Float32Array([7, -3, 2]), new Float32Array([4, 4, -5]), new Float32Array([9, 0, 2])];

export default function Model() {
    const mountRef = useRef<HTMLDivElement>(null);
    const [loading, setLoading] = useState(true);
    const [results, setResults] = useState(null);
    
    

    // Load and run the ONNX model
    useEffect(() => {
        if (isLoaded) {
            return;
        }

        isLoaded = true;
        
        for (let i = 0; i < 5; i++) {
            snakePopulation.push(new Snake(3, 10))
        }

        const s = new Snake(3, 10);
        s.update();
        
        /*snake.loadInferenceSession("/model.onnx")
        .then(() => setInterval(async () => {
            await snake.update([new Float32Array([-9, 3, 3])])
        }, 2000));*/
    }, []);

    

    const beginSimulation = async () => {
        // Set up the camera
        let camera = new THREE.OrthographicCamera();
        camera.position.set(40, 10, 30);
        camera.lookAt(0, 0, 0);
        camera.zoom = 0.1;
        camera.updateProjectionMatrix();

        // Set up the scene and renderer
        let scene = new THREE.Scene();
        const renderer = new THREE.WebGLRenderer();
        renderer.setSize(window.innerWidth, window.innerHeight);
        mountRef.current?.appendChild(renderer.domElement);

        // Declare geometry and mesh materials
        const geometry = new THREE.BoxGeometry();
        const snakeMaterialHead = new THREE.MeshBasicMaterial({ color: 0x72e66c });
        const snakeMaterialTail = new THREE.MeshBasicMaterial({ color: 0x8aace3 })
        const foodMaterial = new THREE.MeshBasicMaterial({ color: 0xfaf561 });
        
        // Enable user camera controls
        const controls = new OrbitControls( camera, renderer.domElement);
        controls.update();
        
        // Load all meshes for the snakes and add it to the snake group
        let snakeGroup = new THREE.Group();
        for (let i = 0; i < snakePopulation.length; i++) {
            // Load the snake's head's mesh and add it to the snake group
            const currentSnake: Snake = snakePopulation[i];
            const snakeMeshHead = new THREE.Mesh(geometry, snakeMaterialTail);
            const currentSnakePosition = [...currentSnake.snakePosition];
            loadMesh(currentSnake.numberOfDimensions, snakeMeshHead, currentSnakePosition)
            // @ts-ignore
            snakeMeshHead.userData.sessionId = currentSnake.session.handler.sessionId;
            console.log(snakeMeshHead.userData.sessionId)
            snakeGroup.add(snakeMeshHead);

            // Load all the snake's tail meshes and add them to the head's mesh's children
            const currentHistory = [...currentSnake.snakeHistory]
            for (let j = 0; j < currentHistory.length / currentSnake.numberOfDimensions; j++) {
                const currentPos = currentHistory.slice(j * currentSnake.numberOfDimensions, (j + 1) * currentSnake.numberOfDimensions);
                const snakeMeshTail = new THREE.Mesh(geometry, snakeMaterialTail);
                loadMesh(currentSnake.numberOfDimensions, snakeMeshTail, currentPos);
                snakeMeshHead.add(snakeMeshTail)
            }
        }
        // Add the snakes to the scene
        scene.add(snakeGroup);

        // Animation loop
        const animate = async () => {
            requestAnimationFrame(animate);
            await Promise.all(snakePopulation.map(async (snake: Snake) => {
                // Forward pass each snake inference model
                await snake.update(foodPositions);
            }))
            .then(() => snakeGroup.children.map((mesh: THREE.Object3D<THREE.Object3DEventMap>, index: number) => {
                // Update mesh positions
                const position = [...snakePopulation[index].snakePosition];
                mesh.position.set(position[0], position[1], position[2]);
            }))  
            .finally(() => renderer.render(scene, camera))
        }

        animate();
    }

    return (
        <div className="w-screen h-screen absolute bg-white z-[1]">
            <div className='z-0 absolute' ref={mountRef}></div>
        </div>
    );
}