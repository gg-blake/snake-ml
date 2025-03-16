'use client';

import React, { useEffect, useState, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import * as tf from '@tensorflow/tfjs';
import { batchArraySync, DifferentialEvolutionTrainer, Data, DataValues, arraySync, calculateNearbyBounds, projectDirectionToBounds, dispose, trainer, ModelState } from './lib/model';
import '@tensorflow/tfjs-backend-webgl';
import Renderer from './lib/renderer';
import { Skeleton } from "../components/ui/skeleton"
import { ThemeProvider } from "../components/theme-provider"
import { SettingsPane, settings, Settings } from './settings';
import getModel from "./lib/layermodel";


import {
    Menubar,
    MenubarContent,
    MenubarItem,
    MenubarMenu,
    MenubarSeparator,
    MenubarShortcut,
    MenubarTrigger,
} from "../components/ui/menubar"

const Loading = () => {
    return (
        <div className='flex absolute w-screen h-screen justify-center items-center gap-4'>
            <Skeleton className='w-[100px] h-[100px] rounded-full' />
            <Skeleton className='w-[100px] h-[100px] rounded-full' />
            <Skeleton className='w-[100px] h-[100px] rounded-full' />
        </div>
    )
}



const TensorFlowModel: React.FC = () => {
    const containerRef = useRef<HTMLDivElement>(null);
    const [isTfReady, setIsTfReady] = useState(false);
    const [isStarted, setIsStarted] = useState(false);
    const killRequest = useRef<boolean>(false);
    const renderer = useRef<Renderer | null>(null);
    const controls = useRef<OrbitControls | null>(null);

    useEffect(() => {
        console.log("page loaded")
        let isMounted = true;
    
        const initializeTf = async () => {
            await tf.setBackend('webgl');
        
            if (isMounted) {
                setIsTfReady(true);
            }
        };
    
        initializeTf();
    
        return () => {
          isMounted = false; // Cleanup in case component unmounts
        };
    }, []);

    const setup = () => {
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl2') as WebGL2RenderingContext;
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const r = new THREE.WebGLRenderer({
            canvas,
            context: gl,
        });
        r.setSize(window.innerWidth, window.innerHeight);
        containerRef.current?.appendChild(r.domElement);
        camera.position.z = 5;

        renderer.current = new Renderer(scene, r, camera);

        const ambientLight = new THREE.AmbientLight(0x404040); // soft white light
        const mainLight = new THREE.DirectionalLight('white', 1);
        mainLight.position.set(10, 10, 10);
        renderer.current.scene.add(ambientLight);
        renderer.current.scene.add(mainLight);

        // Enable user camera controls
        controls.current = new OrbitControls(camera, r.domElement);
        controls.current.update();
        controls.current.target.set(0, 0, 0);
    }

    const train = (renderer: Renderer, controls: OrbitControls) => {
        const B = settings.model.B;
        const T = settings.model.T;
        const C = settings.model.C;
        const CR = settings.trainer.CR;
        const F = settings.trainer.F;
        const G = settings.model.boundingBoxLength;
        const TTL = settings.model.TTL;

        // Initialize Models
        const model = getModel(settings.model);
        
        const targetPositions = tf.randomUniformInt([B, C], -G, G, 'float32');

        // Initialize Renderer
        const state: Data<tf.Variable> = {
            position: tf.variable(tf.zeros([B, C], 'float32')),
            direction: tf.variable(tf.eye(C, ...[,,], 'float32').expandDims(0).tile([B, 1, 1])),
            fitness: tf.variable(tf.ones([B], 'float32').mul(TTL)),
            target: tf.variable(tf.zeros([B], 'int32')),
            history: tf.variable(tf.zeros([B, 1, C], 'float32')),
            fitnessDelta: tf.variable(tf.zeros([B], 'float32')),
            active: tf.variable(tf.ones([B], 'int32'))
        }
        const stateArray = arraySync(state);
        renderer.addBatchParams(stateArray);

        // Add all the target meshes to the scene
        const targets = targetPositions.gather(state.target).arraySync() as number[][];
        renderer.addGroup(targets, Renderer.primaryFoodMaterial);

        // Cube Geometry
        const geometry = new THREE.BoxGeometry(G * 2, G * 2, G * 2);
        // Wireframe Material
        const wireframeMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true });
        // Create Cube Mesh
        const boundBox = new THREE.Mesh(geometry, wireframeMaterial);
        renderer.scene.add(boundBox);

        tf.tidy(() => {
            const proj = projectDirectionToBounds(state.position, state.direction, G);
            const projectionPositions = (proj.arraySync() as number[][]);
            renderer.addGroup(projectionPositions, Renderer.debugMaterial);
            renderer.updateGroupMaterial(1, projectionPositions.map(() => Renderer.debugMaterial));
            console.log(projectionPositions)
        })

        tf.tidy(() => {
            const positionTile = state.position.expandDims(1).tile([1, C, 1]).reshape([B * C, C]);
            const directionTile = positionTile.add(state.direction.reshape([B * C, C]));
            renderer.addLineGroup(positionTile.arraySync() as number[][], directionTile.arraySync() as number[][]);
        })

        renderer.render();

        function animate() {
            if (killRequest.current) {
                killRequest.current = false;
                renderer.resetScene();
                const ambientLight = new THREE.AmbientLight(0x404040); // soft white light
                const mainLight = new THREE.DirectionalLight('white', 1);
                renderer.scene.add(ambientLight);
                renderer.scene.add(mainLight);

                setIsStarted(false);
                return false
            };

            if (state.active.equal(0).all().arraySync() == 1) {
                // Initialize population
            }

            const nextState: tf.Tensor[] = model.predict([
                state.position, 
                state.direction, 
                state.fitness, 
                targetPositions, 
                state.target, 
                state.history, 
                state.active]) as tf.Tensor[]

            state.position.assign(nextState[0] as tf.Tensor2D);
            state.direction.assign(nextState[1] as tf.Tensor3D);
            state.fitness.assign(nextState[2] as tf.Tensor1D);
            state.target.assign(nextState[3] as tf.Tensor1D);
            state.history.assign(nextState[4] as tf.Tensor3D);
            state.fitnessDelta.assign(nextState[5] as tf.Tensor1D);
            state.active.assign(nextState[6] as tf.Tensor1D);

            tf.tidy(() => {
                const proj = projectDirectionToBounds(state.position, state.direction, G);
                const projectionPositions = (proj.arraySync() as number[][]);
                renderer.updateGroupPosition(1, projectionPositions, 1);

                //proj.print()
            })

            tf.tidy(() => {
                const positionTile = state.position.expandDims(1).tile([1, C, 1]).reshape([B * C, C]);
                const directionTile = positionTile.add(state.direction.reshape([B * C, C]));
                renderer.updateLineGroupPosition(2, positionTile.arraySync() as number[][], directionTile.arraySync() as number[][]);
            })

            tf.tidy(() => {
                const targetIndices = state.target.max(0).arraySync() as number;
                renderer.updateGroupMaterial(0, Array.from(Array(T).keys(), (i: number) => i == targetIndices ? Renderer.primaryFoodMaterial : Renderer.secondaryFoodMaterial))
            })

            renderer.updateParams(arraySync(state));
            controls.update();
            renderer.render();
            
            requestAnimationFrame(animate);
        }
    }

    const preAnimate = (...args: ModelState<DifferentialEvolutionTrainer>): boolean => {
        

        const CR = settings.trainer.CR;
        const F = settings.trainer.F;

        if (args[0].state.active.sum().arraySync() === 0 && args[1].state.active.sum().arraySync() === 0) {
            args[0].buildFromCompare(args[1]);
            args[1].buildFrom(args[0], CR, F);
            args[0].resetState();
            args[1].resetState();
        };

        return true;
    }



    const startTraining = () => {
        if (!renderer.current || !controls.current) {
            setup();
        }
        if (!renderer.current || !controls.current) return;
        setIsStarted(true);

        train(renderer.current, controls.current);
    }

    const endTraining = () => {
        if (isStarted) {
            killRequest.current = true;
        }
    }

    return (
        <ThemeProvider
            attribute="class"
            defaultTheme="dark"
            enableSystem
            disableTransitionOnChange
        >
        <SettingsPane />
        {
            isTfReady ?
                <>
                    <Menubar className='absolute m-2'>
                        <MenubarMenu>
                            <MenubarTrigger>Menu</MenubarTrigger>
                            <MenubarContent>
                                <MenubarItem onClick={!isStarted ? () => startTraining() : () => endTraining()}>
                                    {!isStarted ? "Start Training" : "End Training"} <MenubarShortcut>Ctrl+S</MenubarShortcut>
                                </MenubarItem>
                                <MenubarItem>Reset Training<MenubarShortcut>Ctrl+R</MenubarShortcut></MenubarItem>
                                <MenubarSeparator />
                                <MenubarItem>Load Checkpoint</MenubarItem>
                                <MenubarSeparator />
                                <MenubarItem onClick={() => console.log(settings)}>Settings</MenubarItem>
                            </MenubarContent>
                        </MenubarMenu>
                    </Menubar>
                    <div ref={containerRef} />
                </>
                :
                <Loading />
        }
        </ThemeProvider>
    );
};
export default TensorFlowModel;