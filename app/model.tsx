'use client';

import React, { useEffect, useState, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import * as tf from '@tensorflow/tfjs';
import { batchArraySync, DifferentialEvolutionTrainer, Data, DataValues, arraySync, calculateNearbyBounds, projectDirectionToBounds, dispose } from './lib/model';
import '@tensorflow/tfjs-backend-webgl';
import Renderer from './lib/renderer';
import { Skeleton } from "../components/ui/skeleton"
import { ThemeProvider } from "../components/theme-provider"
import { SettingsPane, settings, Settings } from './settings';


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
    const pendingSettings = useRef<Settings>(settings);

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
    }

    const start = (activeSettings: Settings) => {
        
        if (!(typeof window !== 'undefined' && isTfReady && !isStarted)) return;

        setIsStarted(true);
        // Create the scene if none exists
        if (renderer.current === null) {
            setup();
        }

        const B = activeSettings["Batch Size"];
        const T = activeSettings["Number of Food"];
        const C = activeSettings["Number of Dimensions"];
        const CR = activeSettings["Crossover Probabilty"];
        const F = activeSettings["Differential Weight"];
        const TTL = activeSettings["Time To Live"];
        const G = activeSettings["Bound Box Length"];

        // Initialize models
        const currentModel = new DifferentialEvolutionTrainer({ batchInputShape: [B, T, C] });
        const nextModel = new DifferentialEvolutionTrainer({ batchInputShape: [B, T, C] });
        currentModel.build();
        nextModel.build();
        nextModel.buildFrom(currentModel, CR, F);

        // Initialize the game states for both models
        let currentModelState: Data<tf.Tensor> = currentModel.resetInput;
        let nextModelState: Data<tf.Tensor> = nextModel.resetInput;

        // Add all the model meshes to the scene
        const currentModelStateArray = arraySync(currentModelState);
        const nextModelStateArray = arraySync(nextModelState);
        if (renderer.current === null) return
        renderer.current.addBatchParams(currentModelStateArray);
        renderer.current.addBatchParams(nextModelStateArray, B);

        // Add all the target meshes to the scene
        const targets = currentModel.wTarget!.read().arraySync() as number[][];
        renderer.current.addGroup(targets, Renderer.primaryFoodMaterial);

        // Enable user camera controls
        const controls = new OrbitControls(renderer.current.camera, renderer.current.renderer.domElement);
        controls.update();
        controls.target.set(0, 0, 0);
        
        // Cube Geometry
        const geometry = new THREE.BoxGeometry(G*2, G*2, G*2);
        // Wireframe Material
        const wireframeMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true });
        // Create Cube Mesh
        const boundBox = new THREE.Mesh(geometry, wireframeMaterial);
        renderer.current.scene.add(boundBox);

        tf.tidy(() => {
            const proj = projectDirectionToBounds(currentModelState.position, currentModelState.direction, G);
            const projectionPositions = (proj.arraySync() as number[][]);
            renderer.current!.addGroup(projectionPositions, Renderer.debugMaterial);
            renderer.current!.updateGroupMaterial(1, projectionPositions.map(() => Renderer.debugMaterial));
            console.log(projectionPositions)
        })

        tf.tidy(() => {
            const positionTile = currentModelState.position.expandDims(1).tile([1, C, 1]).reshape([B * C, C]);
            const directionTile = positionTile.add(currentModelState.direction.reshape([B * C, C]));
            renderer.current!.addLineGroup(positionTile.arraySync() as number[][], directionTile.arraySync() as number[][]);
        })

        renderer.current.render();
        

        const renderScene = () => {
            if (renderer.current === null) return
            if (killRequest.current) {
                killRequest.current = false;
                dispose(currentModelState);
                dispose(nextModelState);
                renderer.current.resetScene();
                const ambientLight = new THREE.AmbientLight(0x404040); // soft white light
                const mainLight = new THREE.DirectionalLight('white', 1);
                renderer.current.scene.add(ambientLight);
                renderer.current.scene.add(mainLight);
                
                
                setIsStarted(false);
                return;
            };

            if (currentModelState.active.sum().arraySync() === 0 && nextModelState.active.sum().arraySync() === 0) {
                currentModel.buildFromCompare(currentModelState, nextModelState, nextModel);
                nextModel.buildFrom(currentModel, CR, F);
                currentModelState = currentModel.resetInput;
                nextModelState = nextModel.resetInput;
                //console.log("Reset")
            };
            //await tf.nextFrame()
            requestAnimationFrame(renderScene);
            
            
            //console.time("renderScene");
            //console.log(currentModelState);
            
            
            try {
                currentModelState = currentModel.call(currentModelState);
                nextModelState = nextModel.call(nextModelState);
                
            } catch (e) {
                killRequest.current = true;
                throw e;
            }
            
            //currentModelState[2].lessEqual(0).all().print()


            tf.tidy(() => {
                const proj = projectDirectionToBounds(currentModelState.position, currentModelState.direction, G);
                const projectionPositions = (proj.arraySync() as number[][]);
                renderer.current!.updateGroupPosition(1, projectionPositions, 1);

                //proj.print()
            })

            tf.tidy(() => {
                const positionTile = currentModelState.position.expandDims(1).tile([1, C, 1]).reshape([B * C, C]);
                const directionTile = positionTile.add(currentModelState.direction.reshape([B * C, C]));
                renderer.current!.updateLineGroupPosition(2, positionTile.arraySync() as number[][], directionTile.arraySync() as number[][]);
            })

            tf.tidy(() => {
                const targetIndices = currentModelState.target.max(0).arraySync() as number;
                renderer.current!.updateGroupMaterial(0, Array.from(Array(T).keys(), (i: number) => i == targetIndices ? Renderer.primaryFoodMaterial : Renderer.secondaryFoodMaterial))
            })

            renderer.current.updateParams(arraySync(currentModelState));
            renderer.current.updateParams(arraySync(nextModelState), B);
            controls.update();
            renderer.current.render();
            //console.timeEnd("renderScene");
            
        }

        renderScene();
    }

    const end = () => {
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
                                <MenubarItem onClick={!isStarted ? () => start(settings) : () => end()}>
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