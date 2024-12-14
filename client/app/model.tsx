import React, { useEffect, useState } from 'react';
import * as ort from 'onnxruntime-web';

const TTL = 100;
const INITIAL_SCORE = 5;


function createIdentityMatrix(n: number): Float32Array {
    const identityMatrix = new Float32Array(n * n);
    for (let i = 0; i < n; i++) {
        identityMatrix[i * n + i] = 1; // Set the diagonal elements to 1
    }
    return identityMatrix;
}

class SnakeOutputData implements ort.InferenceSession.OnnxValueMapType {
    "next_position": ort.Tensor;
    "velocity": ort.Tensor;
    "nearby_food": ort.Tensor;
    "nearby_body": ort.Tensor;
    "nearby_bounds": ort.Tensor;
}

class Snake {
    numberOfDimensions: number;
    snakePosition: Float32Array;
    snakeHistory: Float32Array;
    snakeBounds: BigInt64Array;
    snakeVelocity: Float32Array;
    snakeScore: number;
    session: ort.InferenceSession | undefined;

    constructor(numberOfDimensions: number, boundsSquareSize: number) {
        this.snakeScore = INITIAL_SCORE;
        this.numberOfDimensions = numberOfDimensions;
        this.snakePosition = new Float32Array([ -1, 6, -1 ]);  // Shape (1, 4)
        this.snakeHistory = new Float32Array([...Array.from({ length: this.snakeScore * numberOfDimensions}, () => 0)]);  // Shape (8, 4)
        this.snakeBounds = new BigInt64Array([...Array.from({ length : this.numberOfDimensions }, () => BigInt(boundsSquareSize))]);  // Shape (1, 4)
        this.snakeVelocity = createIdentityMatrix(numberOfDimensions);
        this.snakeIsAlive = true;
    }

    async loadInferenceSession(filepath: string) {
        try {
            this.session = await ort.InferenceSession.create(filepath);
        } catch (error) {
            console.error(`Error loading inference session: ${error}`)
        }
    }

    async update(foodPositions: Float32Array[]) {
        if (!this.session) {
            console.error("No inference session found");
            return;
        }
        
        if (!this.snakeIsAlive) {
            return;
        }

        const input = {
            'snake_pos': new ort.Tensor('float32', this.snakePosition, [this.numberOfDimensions]),
            'snake_history': new ort.Tensor('float32', this.snakeHistory, [this.snakeScore, this.numberOfDimensions]),
            'food_pos': new ort.Tensor('float32', foodPositions[this.snakeScore - INITIAL_SCORE], [this.numberOfDimensions]),
            'bounds': new ort.Tensor('int64', this.snakeBounds, [this.numberOfDimensions]),
            'vel': new ort.Tensor('float32', this.snakeVelocity, [this.numberOfDimensions, this.numberOfDimensions]),
        };

        try {
            const output: ort.InferenceSession.OnnxValueMapType = await this.session.run(input);
            this.gameUpdate(output);
            
            
        } catch (error) {
            console.error(`There was an error running the model: ${error}`)
        }
    }
    
    gameUpdate(output: ort.InferenceSession.OnnxValueMapType) {
        this.snakePosition = output.next_position.cpuData;
        this.snakeVelocity = output.velocity.cpuData;
        console.log(output.nearby_food);
    }
}


export default function Model() {
    const [loading, setLoading] = useState(true);
    const [results, setResults] = useState(null);
    
    

    // Load and run the ONNX model
    useEffect(() => {

        let snake = new Snake(3, 10);
        snake.loadInferenceSession("/model.onnx")
        .then(() => setInterval(async () => {
            await snake.update([new Float32Array([-9, 3, 3])])
        }, 2000));
    }, []);

    return (
        <div>
        <h1>3D Snake Model Inference</h1>
        {loading ? (
            <p>Loading model...</p>
        ) : (
            <div>
            <h2>Results</h2>
            {results ? (
                <>
                <h3>Next Position:</h3>
                <pre>{JSON.stringify(results['next_position'].data, null, 2)}</pre>

                <h3>Velocity:</h3>
                <pre>{JSON.stringify(results['velocity'].data, null, 2)}</pre>
                </>
            ) : (
                <p>Model could not run successfully</p>
            )}
            </div>
        )}
        </div>
    );
}