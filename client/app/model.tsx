 import React, { useEffect, useState, useRef } from 'react';
import * as ort from 'onnxruntime-web';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { resolve } from 'path';
import RenderResult from 'next/dist/server/render-result';
import { forwardBatch, unitTest } from './lib/modelV2';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';

const TTL = 100;
const INITIAL_SCORE = 5;


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