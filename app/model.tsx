'use client';

import React, { useEffect, useState, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import * as tf from '@tensorflow/tfjs';
import { Data, DataValues, arraySync } from './lib/model';
import '@tensorflow/tfjs-backend-webgl';
import {NEATRenderer} from './lib/renderer';
import { Skeleton } from "../components/ui/skeleton"
import { ThemeProvider } from "../components/theme-provider"
import { SettingsPane, settings, pendingSettings, Settings, stats } from './settings';
import getModel from "./lib/model";
import { toast } from "sonner";




import {
    Menubar,
    MenubarContent,
    MenubarItem,
    MenubarMenu,
    MenubarSeparator,
    MenubarShortcut,
    MenubarTrigger,
} from "../components/ui/menubar"
import NEAT from './lib/optimizer';

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
    const endRequest = useRef<boolean>(false);
    const renderer = useRef<NEATRenderer | null>(null);
    const controls = useRef<OrbitControls | null>(null);
    const trainerRef = useRef<NEAT | null>(null);
    const modelRef = useRef<tf.LayersModel | null>(null);
    const [checkpoint, setCheckpoint] = useState<tf.Tensor3D[] | null>(null);

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

        renderer.current = new NEATRenderer(scene, r, camera);

        const ambientLight = new THREE.AmbientLight(0x404040); // soft white light
        const mainLight = new THREE.DirectionalLight('white', 1);
        mainLight.position.set(10, 10, 10);
        renderer.current.scene.add(ambientLight);
        renderer.current.scene.add(mainLight);


        // Enable user camera controls
        trainerRef.current = new NEAT(settings.model, settings.trainer);
    }

    const train = (model: tf.LayersModel, trainer: NEAT, renderer: NEATRenderer) => {
        const B = settings.model.B;
        const T = settings.model.T;
        const C = settings.model.C;
        const CR = settings.trainer.CR;
        const F = settings.trainer.F;
        const G = settings.model.boundingBoxLength;
        const TTL = settings.model.TTL;

        if (modelRef.current?.feedInputShapes[0][0] == 1) {
            let batchWeights: tf.Tensor3D[] = [];
            (model.getWeights(true) as tf.Tensor3D[]).map((weights: tf.Tensor3D) => {
                batchWeights.push(weights.tile([settings.model.B, 1, 1]));
            })
            model = getModel(settings);
            model.setWeights(batchWeights);
            if (settings.model.B > 1) {
                trainer.evolve(model);
            }
            
            toast("Checkpoint detected. Loading from existing weights.");
        }

        renderer.renderInit(trainer);

        function animate() {
            if (endRequest.current) {
                renderer.resetScene();
                const ambientLight = new THREE.AmbientLight(0x404040); // soft white light
                const mainLight = new THREE.DirectionalLight('white', 1);
                mainLight.position.set(10, 10, 10);
                renderer.scene.add(ambientLight);
                renderer.scene.add(mainLight);
                

                // Reset model and trainer
                const index = trainer.state.fitness.argMax().arraySync() as number;
                tf.tidy(() => {
                    let bestWeights: tf.Tensor3D[] = []; // (1, X, Y)
                    (model.getWeights(true) as tf.Tensor3D[]).map((weights: tf.Tensor3D) => {
                        bestWeights.push(tf.keep(weights.slice([index, 0, 0], [1, weights.shape[1], weights.shape[2]])));
                    });
                    const compactModel = getModel({...settings, model: {
                        ...settings.model,
                        B: 1
                    }});
                    compactModel.setWeights(bestWeights);
                    model.save('downloads://my-model')
                    .then(() => toast("Checkpoint saved successully to local storage"));
                })
                
                trainerRef.current?.resetState();
                
                endTraining();

                return;
            }

            if (killRequest.current) {
                trainer.resetState();
                killRequest.current = false;
            };

            
            const start = Date.now();
            
            if (trainer.epochCount % pendingSettings.renderer.renderEpochInterval == 0 && trainer.epochCount > 30) {
                trainer.step(model, renderer);
            } else {
                trainer.step(model);
            }
            
            const end = Date.now();

            const maxFitness = trainer.state.fitness.max().arraySync() as number;
            if (stats.maxFitnessGlobal < maxFitness) {
                stats.maxFitnessGlobal = maxFitness;
            }

            stats.maxFitness = maxFitness;
            stats.framesToRestart = trainer.timeToRestart.arraySync();
            stats.fps = 1 / ((end - start) / 1000);
            stats.timeToRestart = stats.fps > 0 ? stats.framesToRestart / stats.fps : 0;
            stats.epochCount = trainer.epochCount;
            
            requestAnimationFrame(animate);
        }

        animate();
    }

    const startTraining = () => {
        if (!renderer.current || !trainerRef.current || !modelRef.current) {
            setup();
        }
        
        modelRef.current = getModel(settings);
        setIsStarted(true);
    }

    useEffect(() => {
        if (!isStarted) return;
        if (!renderer.current || !trainerRef.current || !modelRef.current) return;
        endRequest.current = false;
        tf.loadLayersModel('downloads://my-model')
        .then((model: tf.LayersModel) => {
            modelRef.current = model;
            console.log(modelRef.current)
            toast("Model loaded from cache successfully");
            train(modelRef.current, trainerRef.current!, renderer.current!);
        })
        .catch((reason: any) => {
            toast("No model detected. Starting with new weights.")
            train(modelRef.current!, trainerRef.current!, renderer.current!);
        })
    }, [isStarted])

    const endTraining = () => {
        if (!isStarted) return;
        if (isStarted) {
            endRequest.current = true;
            Object.assign(settings, pendingSettings);
            setIsStarted(false);
        }
    }

    const killPopulation = () => {
        killRequest.current = true;
    }

    useEffect(() => {
        toast("Checkpoint saved successfully");
    }, [checkpoint])

    return (
        <ThemeProvider
            attribute="class"
            defaultTheme="dark"
            enableSystem
            disableTransitionOnChange
        >
        <SettingsPane killRequest={killRequest} endRequest={endRequest} />
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
                                <MenubarItem onClick={isStarted ? () => killPopulation() : () => {}}>Start Next Generation<MenubarShortcut>Ctrl+R</MenubarShortcut></MenubarItem>
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