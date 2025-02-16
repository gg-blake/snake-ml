'use client';

import React, { useEffect, useState, useRef, LegacyRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Pane } from "tweakpane";

import "@/app/globals.css";
import { resolve } from 'path';
import { use } from "react";
import { Suspense } from "react";
import { Trainer, SnakeModel, NDGameObject, NDVector } from './lib/model';


function loadMesh(mesh: THREE.Mesh<THREE.BoxGeometry, THREE.MeshBasicMaterial, THREE.Object3DEventMap>, position: number[]) {
    mesh.position.set(position[0], position[1], position.length > 2 ? position[2] : 0)
}


var trainer = new Trainer(3, 10, 0.9, 0.8);

export default function NativeClient() {
    const mountRef = useRef<HTMLDivElement>(null); // Reference to mount the 3D scene
    const [presets, setPresets] = useState<Presets>(defaultPreset)
    const [viewPresets, setViewPresets] = useState<ViewPresets>(defaultViewPreset)
    const data = useRef<StreamResponse|null>(null);

    const initGame = () => {

        // Set up the camera
        const aspect = window.innerWidth / window.innerHeight;
        const frustumSize = 100; // Controls how "zoomed in" the scene appears

        const camera = new THREE.OrthographicCamera(
            -frustumSize * aspect / 2,  // left
            frustumSize * aspect / 2,   // right
            frustumSize / 2,            // top
            -frustumSize / 2,           // bottom
            0,                        // near
            1000                        // far
        );
        camera.position.set(500, 500, 500)
        

        // Set up the scene and renderer
        let scene = new THREE.Scene();
        const renderer = new THREE.WebGLRenderer({
        logarithmicDepthBuffer: true 
        });
        renderer.setSize(window.innerWidth, window.innerHeight);
        mountRef.current?.appendChild(renderer.domElement);

        // Declare geometry and mesh materials
        const geometry = new THREE.BoxGeometry();
        const snakeMaterialHead = new THREE.MeshBasicMaterial({ color: 0x72e66c });
        const snakeMaterialTail = new THREE.MeshBasicMaterial({ color: 0x8aace3 });
        const snakeMaterialDead = new THREE.MeshBasicMaterial({ color: 0x1a1a1a });
        const foodMaterial = new THREE.MeshBasicMaterial({ color: 0x403f23 });

        // Enable user camera controls
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.update();
        controls.target.set(0, 0, 0);

        // Scene lighting
        const light = new THREE.DirectionalLight(0xfffaff, 20);
        light.position.set(30, 3, 3);
        light.castShadow = true; // Allow light to cast shadows
        scene.add(light);

        
        
        /* To ensure that all the snake tail meshes are mapped to its head during render properly, 
        I decided to implement a lookup table that maps the snake uid from the server to mesh uuids, 
        I also mapped each uid to the list of its tail mesh uuids */
        let uidLookup: {[key: number]: string} = {} // Head mesh uuid lookup table
        let uidLookup2: {[key: string]: string} = {} // Tail mesh uuid lookup table

        const initSnakeMeshes = (group: THREE.Group<THREE.Object3DEventMap>) => {
        for (let i = 0; i < trainer.currentCandidates.length; i++) {
            // Load the snake's head's mesh and add it to the snake group
            const currentSnake: SnakeModel = trainer.currentCandidates[i];
            const snakeMeshHead = new THREE.Mesh(geometry, snakeMaterialHead);
            const currentSnakePosition = currentSnake.gameObject.position.arraySync();
            loadMesh(snakeMeshHead, currentSnakePosition)
            uidLookup[currentSnake.uid] = snakeMeshHead.uuid
            group.add(snakeMeshHead);

            // Load all the snake's tail meshes and add them to the head's mesh's children
            const currentHistory = currentSnake.history

            for (let j = 0; j < currentHistory.length / presets.n_dims; j++) {
                const currentPos = currentHistory[j];
                const snakeMeshTail = new THREE.Mesh(geometry, snakeMaterialTail);
                uidLookup2[`${currentSnake.uid}-${j}`] = snakeMeshTail.uuid;
                loadMesh(snakeMeshTail, currentPos);
                group.add(snakeMeshTail)
            }
        }

        // Add the snakes to the scene
        scene.add(group);
        }

        // Load all meshes for the snakes and add it to the snake group
        let snakeGroup = new THREE.Group();
        initSnakeMeshes(snakeGroup);

        // Load all meshes for the food and add it to the snake group
        let foodGroup = new THREE.Group();
        scene.add(foodGroup);

        // Render a wireframe bounding box
        if (viewPresets.viewBounds) {
            const boundBoxGeometry = new THREE.BoxGeometry(presets.width * 2, presets.width * 2, presets.n_dims > 2 ? presets.width * 2 : 1); // (width, height, depth)
            const boundBoxWireframe = new THREE.WireframeGeometry(boundBoxGeometry);
            const boundBoxLineSegments = new THREE.LineSegments(boundBoxWireframe, new THREE.LineBasicMaterial({ color: 0xffffff }));
            scene.add(boundBoxLineSegments);
        }
        // Animation loop
        const animate = () => {
            requestAnimationFrame(animate);
            if (!data.current) {
                return
            }

            let snakePopulation = data.current.snake_data;

            if (snakePopulation.every((snake: SnakeData) => !snake.alive || snake.pos.every(p => p == 0))) {
                // Remove object uuids from lookup table
                uidLookup = {};
                uidLookup2 = {};

                // Reset the snake group, removing old snake guts
                scene.remove(snakeGroup);
                // Load all meshes for the snakes and add it to the snake group
                snakeGroup = new THREE.Group();
                initSnakeMeshes(snakeGroup);

                // Reset food
                scene.remove(foodGroup)
                foodGroup = new THREE.Group();
                scene.add(foodGroup);
            }

            // Calculate best snake and display the best snake (feature)
            let maxFitness = 0;
            let bestSnakeIndex = 0;
            if (viewPresets.colorMode == "best") {
            snakePopulation.map((snake: SnakeData, index: number) => {
                if (snake.fitness > maxFitness) {
                maxFitness = snake.fitness;
                bestSnakeIndex = index;
                }
            })
            }

            // Render each snake and all of its assets
            snakePopulation.map((snake: SnakeData, snakeIndex: number) => {
            const currentUID = uidLookup[snake.uid];
            const meshRef = snakeGroup.getObjectByProperty('uuid', currentUID)! as THREE.Mesh<THREE.BoxGeometry, THREE.MeshBasicMaterial, THREE.Object3DEventMap>;

            const materialCopy = meshRef.material.clone();
            if (!snake.alive) {
                meshRef.material = materialCopy;
                meshRef.material.color.lerpHSL(new THREE.Color("hsl(139, 2%, 3%)"), 0.5)
            } else {
                meshRef.material = materialCopy;
                meshRef.material.color.lerpHSL(new THREE.Color("hsl(139, 78%, 63%)"), 0.5)
            }
            
            // Display best snake (feature)
            if (viewPresets.colorMode == "best") {
                if (bestSnakeIndex == snakeIndex) {
                meshRef.material = materialCopy;
                meshRef.material.color.lerpHSL(new THREE.Color("hsl(139, 78%, 63%)"), 0.5)
                } else {
                meshRef.material = materialCopy;
                meshRef.material.color.lerpHSL(new THREE.Color("hsl(139, 46%, 11%)"), 0.5)
                }
            }

            if (presets.n_dims >= 4) {
                meshRef.material = materialCopy;
                meshRef.material.color.lerpHSL(new THREE.Color(`hsl(${((snake.pos[3] + presets.width) / (presets.width * 2)) * 360}, 78%, 63%)`), 0.5);
            }

            for( var i = scene.children.length - 1; i >= 0; i--) { 
                let obj = meshRef.children[i];
                meshRef.remove(obj); 
            }



            const normalMesh = new THREE.Mesh(geometry, snakeMaterialTail);

            const currentSnakePosition = new THREE.Vector3(...snake.pos);
            
            
            
            
            
            const currentSnakeHistory = snake.history;
            meshRef.position.lerp(currentSnakePosition, 1);
            
            currentSnakeHistory.map((pos: number[], index: number) => {
                if (index == 0) {
                return
                }
                let snakeMeshTail: THREE.Mesh<THREE.BoxGeometry, THREE.MeshBasicMaterial, THREE.Object3DEventMap>;
                const key = `${currentUID}-${index}`;
                if (Object.keys(uidLookup2).includes(key)) {
                snakeMeshTail = snakeGroup.getObjectByProperty('uuid', uidLookup2[key])! as THREE.Mesh<THREE.BoxGeometry, THREE.MeshBasicMaterial, THREE.Object3DEventMap>;
                
                } else {
                snakeMeshTail = new THREE.Mesh(geometry, snakeMaterialTail);
                uidLookup2[`${currentUID}-${index}`] = snakeMeshTail.uuid;
                loadMesh(snakeMeshTail, pos);
                snakeGroup.add(snakeMeshTail)
                }

                snakeMeshTail.material = meshRef.material;
                snakeMeshTail.position.lerp(new THREE.Vector3(...pos), 1);
            })

            
            })

            
            const foodPositions = data.current.food_data;
            foodPositions.map((pos: number[], index: number) => {
            const currentFoodMesh = new THREE.Mesh(geometry, foodMaterial);
            // Highlight the most recent food
            if (index == foodPositions.length - 1) {
                currentFoodMesh.material = currentFoodMesh.material.clone();
                currentFoodMesh.material.color.setHex(0xeee617);
            } else {
                currentFoodMesh.material = currentFoodMesh.material.clone();
                currentFoodMesh.material.color.setHex(0x403f23);
            }

            loadMesh(currentFoodMesh, pos);
            foodGroup.add(currentFoodMesh);
            });

            // Render updated snake positions
            renderer.render(scene, camera);
            
        }

        animate();
        
    }

    useEffect(() => {
        const eventSource = new EventSource('http://localhost:8000/stream');

        eventSource.onmessage = (event) => {
        data.current = JSON.parse(event.data);
        if (!(mountRef.current && mountRef.current.hasChildNodes())) {
            initGame();
        }
        }

        eventSource.onerror = (error) => {
        console.error('EventSource error:', error);
        eventSource.close();
        };

        return () => {
        eventSource.close();
        };
    }, []);

    return (
        <div className="w-screen h-screen absolute bg-black z-[1]">
        <div className='z-0 absolute' ref={mountRef}></div>
        <TweakPaneMenu params={presets} viewParams={viewPresets} setParams={setPresets} setViewParams={setViewPresets} />
        </div>
    );
    }
