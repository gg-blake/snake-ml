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

    const initializer = (renderer: Renderer, controls: OrbitControls): ModelState<DifferentialEvolutionTrainer> => {
        const B = settings["Batch Size"];
        const T = settings["Number of Food"];
        const C = settings["Number of Dimensions"];
        const CR = settings["Crossover Probabilty"];
        const F = settings["Differential Weight"];
        const G = settings["Bound Box Length"];

        // Initialize Models
        const currentModel = new DifferentialEvolutionTrainer({ batchInputShape: [B, T, C] });
        const nextModel = new DifferentialEvolutionTrainer({ batchInputShape: [B, T, C] });
        
        currentModel.build();
        nextModel.build();
        nextModel.buildFrom(currentModel, CR, F);

        // Initialize Renderer
        const currentStateArray = arraySync(currentModel.state);
        const nextStateArray = arraySync(nextModel.state);
        renderer.addBatchParams(currentStateArray);
        renderer.addBatchParams(nextStateArray, B);

        // Add all the target meshes to the scene
        const targets = currentModel.wTarget!.read().arraySync() as number[][];
        renderer.addGroup(targets, Renderer.primaryFoodMaterial);

        // Cube Geometry
        const geometry = new THREE.BoxGeometry(G * 2, G * 2, G * 2);
        // Wireframe Material
        const wireframeMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true });
        // Create Cube Mesh
        const boundBox = new THREE.Mesh(geometry, wireframeMaterial);
        renderer.scene.add(boundBox);

        tf.tidy(() => {
            const proj = projectDirectionToBounds(currentModel.state.position, currentModel.state.direction, G);
            const projectionPositions = (proj.arraySync() as number[][]);
            renderer.addGroup(projectionPositions, Renderer.debugMaterial);
            renderer.updateGroupMaterial(1, projectionPositions.map(() => Renderer.debugMaterial));
            console.log(projectionPositions)
        })

        tf.tidy(() => {
            const positionTile = currentModel.state.position.expandDims(1).tile([1, C, 1]).reshape([B * C, C]);
            const directionTile = positionTile.add(currentModel.state.direction.reshape([B * C, C]));
            renderer.addLineGroup(positionTile.arraySync() as number[][], directionTile.arraySync() as number[][]);
        })

        renderer.render();

        return [
            currentModel,
            nextModel
        ]
    }

    const preAnimate = (...args: ModelState<DifferentialEvolutionTrainer>): boolean => {
        if (!renderer.current) return false; 
        if (killRequest.current) {
            killRequest.current = false;
            renderer.current.resetScene();
            const ambientLight = new THREE.AmbientLight(0x404040); // soft white light
            const mainLight = new THREE.DirectionalLight('white', 1);
            renderer.current.scene.add(ambientLight);
            renderer.current.scene.add(mainLight);

            setIsStarted(false);
            return false
        };

        const CR = settings["Crossover Probabilty"];
        const F = settings["Differential Weight"];

        if (args[0].state.active.sum().arraySync() === 0 && args[1].state.active.sum().arraySync() === 0) {
            args[0].buildFromCompare(args[1]);
            args[1].buildFrom(args[0], CR, F);
            args[0].resetState();
            args[1].resetState();
        };

        return true;
    }

    const postAnimate = (animate: () => void, ...args: ModelState<DifferentialEvolutionTrainer>) => {
        requestAnimationFrame(animate);
        if (!renderer.current || !controls.current) return;

        const B = settings["Batch Size"];
        const T = settings["Number of Food"];
        const C = settings["Number of Dimensions"];
        const G = settings["Bound Box Length"];

        tf.tidy(() => {
            const proj = projectDirectionToBounds(args[0].state.position, args[0].state.direction, G);
            const projectionPositions = (proj.arraySync() as number[][]);
            renderer.current!.updateGroupPosition(1, projectionPositions, 1);

            //proj.print()
        })

        tf.tidy(() => {
            const positionTile = args[0].state.position.expandDims(1).tile([1, C, 1]).reshape([B * C, C]);
            const directionTile = positionTile.add(args[0].state.direction.reshape([B * C, C]));
            renderer.current!.updateLineGroupPosition(2, positionTile.arraySync() as number[][], directionTile.arraySync() as number[][]);
        })

        tf.tidy(() => {
            const targetIndices = args[0].state.target.max(0).arraySync() as number;
            renderer.current!.updateGroupMaterial(0, Array.from(Array(T).keys(), (i: number) => i == targetIndices ? Renderer.primaryFoodMaterial : Renderer.secondaryFoodMaterial))
        })

        renderer.current.updateParams(arraySync(args[0].state));
        renderer.current.updateParams(arraySync(args[1].state), B);
        controls.current.update();
        renderer.current.render();
    }



    const startTraining = () => {
        if (!renderer.current || !controls.current) {
            setup();
        }
        if (!renderer.current || !controls.current) return;
        setIsStarted(true);

        trainer(
            initializer(renderer.current, controls.current),
            preAnimate,
            postAnimate
        );
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
                                <MenubarItem onClick={!isStarted ? () => train() : () => end()}>
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