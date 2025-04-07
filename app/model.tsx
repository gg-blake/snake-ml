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
import { FileUploader } from "react-drag-drop-files";
import * as GL from "./lib/layers/gamelayer";
import { Layer, LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';
import { Minus, Plus } from "lucide-react"

import {
  Drawer,
  DrawerClose,
  DrawerContent,
  DrawerDescription,
  DrawerFooter,
  DrawerHeader,
  DrawerTitle,
  DrawerTrigger,
} from "@/components/ui/drawer";

import Dropzone, { DropzoneState, DropzoneProps } from 'shadcn-dropzone';


import {
    Menubar,
    MenubarContent,
    MenubarItem,
    MenubarMenu,
    MenubarSeparator,
    MenubarShortcut,
    MenubarTrigger,
} from "../components/ui/menubar";

import { Input } from "@/components/ui/input"

import { Button } from "@/components/ui/button";

import NEAT from './lib/optimizer';
import { Preahvihear } from 'next/font/google';
import InputNorm from './lib/layers/inputnorm';

const Loading = () => {
    return (
        <div className='flex absolute w-screen h-screen justify-center items-center gap-4'>
            <Skeleton className='w-[100px] h-[100px] rounded-full' />
            <Skeleton className='w-[100px] h-[100px] rounded-full' />
            <Skeleton className='w-[100px] h-[100px] rounded-full' />
        </div>
    )
}

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

interface ModelCheckpoint {
    json: File | null;
    bin: File | null;
}

const CreateModelForm = ({onDecrease, onIncrease, value, min, max, label}: {onDecrease: () => void, onIncrease: () => void, value: number, min: number, max: number, label: string}) => {
    return (
        <div className="flex flex-row w-auto max-w-full h-auto items-center justify-center p-3 gap-x-3">
            <Button
                variant="outline"
                size="icon"
                className="h-8 w-8 shrink-0 rounded-full"
                onClick={onDecrease}
                disabled={value <= min}
            >
                <Minus />
                <span className="sr-only">Decrease</span>
            </Button>
            <div className="flex-1 text-center">
                <div className="text-7xl font-bold tracking-tighter">
                {value}
                </div>
                <div className="text-[0.70rem] uppercase text-muted-foreground">
                { label }
                </div>
            </div>
            <Button
                variant="outline"
                size="icon"
                className="h-8 w-8 shrink-0 rounded-full"
                onClick={onIncrease}
                disabled={value >= max}
            >
                <Plus />
                <span className="sr-only">Increase</span>
            </Button>
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
    const [checkpoint, setCheckpoint] = useState<ModelCheckpoint>({
        json: null,
        bin: null
    });
    const [modelHyperparams, setModelHyperparams] = useState<GL.Config & LayerArgs>({
        ...settings.model,
        batchInputShape: settings.model.batchInputShape as number[],
    } as GL.Config & LayerArgs);
    const [saveFileName, setSaveFileName] = useState<string>("my-model-1");

    useEffect(() => {
        console.log("page loaded")
        let isMounted = true;
    
        tf.loadLayersModel('localstorage://temporary-model')
        .then(model => {
            const config = (model.getLayer(InputNorm.className) as InputNorm).getConfig();
            setModelHyperparams(config);
            modelRef.current = model;
        })
        .catch((error) => {
            console.log("Checkpoint not found in local storage")
        })

        const initializeTf = async () => {
            await tf.setBackend('webgl');
        
            if (isMounted) {
                setIsTfReady(true);
            }
        };

        
    
        initializeTf();

        window.addEventListener('beforeunload', saveLocal);

        

        return () => {
            window.removeEventListener('beforeunload', saveLocal);
            
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
    }

    const train = () => {
        

        /*if (modelRef.current?.feedInputShapes[0][0] == 1) {
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
        }*/

        if (!renderer.current || !trainerRef.current || !modelRef.current) return;
        renderer.current.renderInit(trainerRef.current);

        function animate() {
            if (!renderer.current || !trainerRef.current || !modelRef.current) return;
            if (endRequest.current) {
                endRequest.current = false;
                modelRef.current = null;
                containerRef.current?.removeChild(renderer.current.renderer.domElement);
                renderer.current = null;
                setIsStarted(false);
                return;
            }

            if (killRequest.current) {
                trainerRef.current.resetState();
                killRequest.current = false;
            };

            
            const start = Date.now();
            
            if (trainerRef.current.epochCount % pendingSettings.renderer.renderEpochInterval == 0) {
                trainerRef.current.step(modelRef.current, renderer.current);
            } else {
                trainerRef.current.step(modelRef.current);
            }
            
            const end = Date.now();

            const maxFitness = trainerRef.current.state.fitness.max().arraySync() as number;
            if (stats.maxFitnessGlobal < maxFitness) {
                stats.maxFitnessGlobal = maxFitness;
            }

            stats.maxFitness = maxFitness;
            stats.framesToRestart = trainerRef.current.timeToRestart.arraySync();
            stats.fps = 1 / ((end - start) / 1000);
            stats.timeToRestart = stats.fps > 0 ? stats.framesToRestart / stats.fps : 0;
            stats.epochCount = trainerRef.current.epochCount;
            
            requestAnimationFrame(animate);
        }

        animate();
    }

    const startTraining = (config: GL.Config & LayerArgs) => {
        settings.model = config;
        if (!modelRef.current) {
            modelRef.current = getModel(settings.model);
        }

        if (!trainerRef.current) {
            trainerRef.current = new NEAT(settings);
        }

        if (!renderer.current) {
            setup();
        }

        setIsStarted(true);
    }

    useEffect(() => {
        if (!isStarted) return;
        if (!renderer.current || !trainerRef.current || !modelRef.current) return;
        endRequest.current = false;
        killRequest.current = false;
        setCheckpoint({ json: null, bin: null });
        train();
    }, [isStarted])

    const killPopulation = () => {
        killRequest.current = true;
    }

    const handleFileUpload = (files: File[]) => {
        files.map((file: File) => {
            if (file.type == "application/json") {
                setCheckpoint((prev: ModelCheckpoint) => ({
                    bin: prev.bin,
                    json: file
                }))
            } else if (file.type == "application/octet-stream") {
                setCheckpoint((prev: ModelCheckpoint) => ({
                    bin: file,
                    json: prev.json
                }))
            }
        })
    }

    useEffect(() => {
        if (checkpoint.json == null || checkpoint.bin == null) return;
        const files = tf.io.browserFiles([checkpoint.json, checkpoint.bin]) as tf.io.IOHandler;
        tf.loadLayersModel(files)
        .then((model => {
            modelRef.current = model;
            const config = (model.getLayer(InputNorm.className) as InputNorm).getConfig();
            setModelHyperparams(config);
            toast("Model loaded from checkpoint", {style: {backgroundColor: "#78cf3e"}});
        }))
        .catch((reason: any) => {
            console.log(reason);
            toast("Failed to load model from checkpoint", {style: {backgroundColor: "#e64040"}});
        })
    }, [checkpoint])

    const adjustModelHyperparams = (value: React.SetStateAction<GL.Config & LayerArgs>): void => {
        if (modelRef.current) {
            modelRef.current = null;
            setCheckpoint({ json: null, bin: null });
        }

        setModelHyperparams(value);
    }

    const ModelCreationMenu = () => {
        return (
            <div className="w-full h-full grid grid-cols-2 grid-rows-[auto,_auto] md:flex-row p-24 z-10 relative gap-y-0 gap-x-3">
                <div className="grid grid-cols-subgrid col-span-2 items-center justify-items-center">
                    <h1 className="w-full h-full text-6xl font-bold border-primary-foreground border-[5px] border-b-0 flex flex-col items-center justify-center gap-3">
                        Create
                        {
                            (checkpoint.json == null || checkpoint.bin == null) &&
                            <Button
                                variant="outline"
                                className="text-4xl px-3 py-6 flex items-center justify-center font-normal"
                                onClick={() => startTraining(modelHyperparams)}
                            >
                                { modelRef.current ? "Continue Training" : "Start Training"}
                                <span className="sr-only">Start Training</span>
                            </Button>
                        }
                    </h1>
                    <h1 className="w-full h-full text-6xl font-bold border-primary-foreground border-[5px] border-b-0 flex flex-col items-center justify-center border-dashed gap-3">
                        Upload
                        {
                            (checkpoint.json != null && checkpoint.bin != null) &&
                            <Button
                                variant="outline"
                                className="text-4xl px-3 py-6 flex items-center justify-center font-normal"
                                onClick={() => startTraining(modelHyperparams)}
                            >
                                Start Training
                                <span className="sr-only">Start Training</span>
                            </Button>
                        }
                    </h1>
                    <div className="absolute p-3 rounded-full border-primary-foreground border-[5px] w-14 h-14 flex items-center justify-center bg-background">or</div>
                </div>
                <div className='border-primary-foreground border-[5px] border-t-0 h-full w-full flex flex-col items-center justify-center overflow-y-auto  rounded-lg relative p-3'>
                    
                    <CreateModelForm
                        onDecrease={() => adjustModelHyperparams((prev: GL.Config & LayerArgs) => ({
                            ...prev,
                            units: prev.units - 1
                        } as GL.Config & LayerArgs)) }
                        onIncrease={() => adjustModelHyperparams((prev: GL.Config & LayerArgs) => ({
                            ...prev,
                            units: prev.units + 1
                        } as GL.Config & LayerArgs)) }
                        value={modelHyperparams.units}
                        min={1}
                        max={300}
                        label="Hidden Layer Size"
                    />
                    <CreateModelForm
                        onDecrease={() => adjustModelHyperparams((prev: GL.Config & LayerArgs) => ({
                            ...prev,
                            batchInputShape: [prev.batchInputShape![0]! - 1, prev.batchInputShape![1], prev.batchInputShape![2]]
                        } as GL.Config & LayerArgs)) }
                        onIncrease={() => adjustModelHyperparams((prev: GL.Config & LayerArgs) => ({
                            ...prev,
                            batchInputShape: [prev.batchInputShape![0]! + 1, prev.batchInputShape![1], prev.batchInputShape![2]]
                        } as GL.Config & LayerArgs)) }
                        value={modelHyperparams.batchInputShape![0] || 0}
                        min={1}
                        max={1000}
                        label="Batch Size"
                    />
                    <CreateModelForm
                        onDecrease={() => adjustModelHyperparams((prev: GL.Config & LayerArgs) => ({
                            ...prev,
                            batchInputShape: [prev.batchInputShape![0], prev.batchInputShape![1]! - 1, prev.batchInputShape![2]]
                        } as GL.Config & LayerArgs)) }
                        onIncrease={() => adjustModelHyperparams((prev: GL.Config & LayerArgs) => ({
                            ...prev,
                            batchInputShape: [prev.batchInputShape![0], prev.batchInputShape![1]! + 1, prev.batchInputShape![2]]
                        } as GL.Config & LayerArgs)) }
                        value={modelHyperparams.batchInputShape![1] || 0}
                        min={5}
                        max={300}
                        label="History Length"
                    />
                    <CreateModelForm
                        onDecrease={() => adjustModelHyperparams((prev: GL.Config & LayerArgs) => ({
                            ...prev,
                            batchInputShape: [prev.batchInputShape![0], prev.batchInputShape![1], prev.batchInputShape![2]! - 1]
                        } as GL.Config & LayerArgs)) }
                        onIncrease={() => adjustModelHyperparams((prev: GL.Config & LayerArgs) => ({
                            ...prev,
                            batchInputShape: [prev.batchInputShape![0], prev.batchInputShape![1], prev.batchInputShape![2]! + 1]
                        } as GL.Config & LayerArgs)) }
                        value={modelHyperparams.batchInputShape![2] || 0}
                        min={2}
                        max={300}
                        label="Spatial Dimensions"
                    />
                    
                    
                </div>
                <div className='border-primary-foreground border-[5px] border-t-0 h-full w-full flex flex-col items-center justify-center border-dashed'>
                    <Dropzone containerClassName='w-full h-full' dropZoneClassName='w-full h-full flex flex-col items-center justify-center text-4xl hover:border-white border-1'
                    onDrop={handleFileUpload}
                    >
                        {(dropzone: DropzoneState) => (
                            <>
                            {
                                dropzone.isDragAccept ? (
                                <div className='text-sm font-medium'>Drop your files here!</div>
                                ) : (
                                <div className='flex items-center flex-col gap-1.5'>
                                    <div className='flex items-center flex-row gap-0.5 text-sm font-medium'>
                                    Upload checkpoint
                                    </div>
                                </div>
                                )
                            }
                            <div className='text-xs text-gray-400 font-medium'>
                                Files left to upload: { checkpoint.json == null && ".json" } { checkpoint.json == null && ".bin" }
                            </div>
                            </>
                        )}
                    </Dropzone>
                </div>
            </div>
        )
    }

    const saveLocal = async () => {
        if (!modelRef.current) return;
        const model = modelRef.current;
        await model.save("localstorage://temporary-model");
        if (!containerRef.current || !renderer.current) return;
        containerRef.current.removeChild(renderer.current.renderer.domElement);
        renderer.current = null;
    }

    const save = async (filename: string) => {
        if (!modelRef.current) return;
        const model = modelRef.current;
        await model.save(`downloads://${filename}`);
        toast("Model saved successfully", {style: {backgroundColor: "#78cf3e"}})
    }

    const handleSave = async (filename: string) => {
        await save(filename)
        handleEnd();
    }

    const handleEnd = async () => {
        endRequest.current = true;
        if (!modelRef.current) return;
        await modelRef.current.save("localstorage://temporary-model");
        window.location.reload();
    }

    return (
        <ThemeProvider
            attribute="class"
            defaultTheme="dark"
            enableSystem
            disableTransitionOnChange
        >
        
        {
            isTfReady && 
            <div className="w-screen h-screen flex items-center justify-center">
                { !isStarted && <ModelCreationMenu /> }
                { (trainerRef.current && modelRef.current) &&
                    <Dialog>
                        <DialogTrigger className='absolute top-3 left-3 z-50'>Open</DialogTrigger>
                        <DialogContent>
                            <DialogHeader>
                                <DialogTitle>Save Checkpoint?</DialogTitle>
                            </DialogHeader>
                            <Input onChange={(e) => setSaveFileName(e.target.value)} />
                            
                            <div className="grid grid-cols-2 gap-3">
                                <Button onClick={() => handleSave(saveFileName)} variant="default" className='w-full h-auto py-3 rounded-lg'>Save</Button>
                                <Button onClick={() => handleEnd()} variant="outline" className="w-full h-auto py-3 rounded-lg">Don't Save</Button>
                            </div>
                            <DialogFooter className="opacity-50 text-primary">
                                Files to be saved: {saveFileName}.json {saveFileName}.bin
                            </DialogFooter>
                        </DialogContent>
                    </Dialog>
                }
                <div className='absolute w-full h-full top-0 left-0 z-0' ref={containerRef} />
            </div>
        }
        {
            isStarted && <SettingsPane killRequest={killRequest} endRequest={endRequest} />
        }
        </ThemeProvider>
    );
};
export default TensorFlowModel;