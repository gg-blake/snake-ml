'use client';

import React, { useEffect, useState, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import * as tf from '@tensorflow/tfjs';
import "@/app/globals.css";
import { TTL, B, T, C, INPUT_LAYER_SIZE } from './lib/constants';
import { batchArraySync, BatchParamsTensor, forwardBatch, compareWeights, generateWeights } from './lib/model';
import '@tensorflow/tfjs-backend-webgl';
import Renderer from './lib/renderer';
import { Skeleton } from "@/components/ui/skeleton"
import { ThemeProvider } from "@/components/theme-provider"

import {
    Menubar,
    MenubarContent,
    MenubarItem,
    MenubarMenu,
    MenubarSeparator,
    MenubarShortcut,
    MenubarTrigger,
} from "@/components/ui/menubar"

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

    const start = () => tf.tidy(() => {
        if (!(typeof window !== 'undefined' && isTfReady && !isStarted)) {
            return
        }

        setIsStarted(true);
        if (renderer.current === null) {
            setup();
        }

        // Model unit test for minimizeBatch()
        let currentParams: BatchParamsTensor = {
            fitness: tf.ones([B]).mul(TTL) as tf.Tensor1D,
            position: tf.zeros([B, C]),
            targetIndices: tf.zeros([B], 'int32'),
            direction: tf.eye(C).expandDims(0).tile([B, 1, 1]),
            history: tf.zeros([B, T, C - 1]).concat(tf.range(0, T).reshape([1, T]).tile([B, 1]).expandDims(-1).neg(), -1) as tf.Tensor3D,
            weights: [tf.keep(tf.randomUniform([B, T, INPUT_LAYER_SIZE])), tf.keep(tf.randomUniform([B, Math.floor((C - 1) * C / 2), T]))],
            alive: tf.ones([B], 'int32')
        }
        let nextParams: BatchParamsTensor = {
            fitness: tf.ones([B]).mul(TTL) as tf.Tensor1D,
            position: tf.zeros([B, C]),
            targetIndices: tf.zeros([B], 'int32'),
            direction: tf.eye(C).expandDims(0).tile([B, 1, 1]),
            history: tf.zeros([B, T, C - 1]).concat(tf.range(0, T).reshape([1, T]).tile([B, 1]).expandDims(-1).neg(), -1) as tf.Tensor3D,
            weights: [tf.keep(tf.randomUniform([B, T, INPUT_LAYER_SIZE])), tf.keep(tf.randomUniform([B, Math.floor((C - 1) * C / 2), T]))],
            alive: tf.ones([B], 'int32')
        };

        const targets: tf.Tensor2D = tf.keep(tf.randomUniform([T, C], -10, 10));

        const currentParamsArray = batchArraySync(currentParams);
        const nextParamsArray = batchArraySync(nextParams);
        if (renderer.current === null) return
        renderer.current.addBatchParams(currentParamsArray);
        renderer.current.addBatchParams(nextParamsArray, B);

        // Enable user camera controls
        const controls = new OrbitControls(renderer.current.camera, renderer.current.renderer.domElement);
        controls.update();
        controls.target.set(0, 0, 0);



        const renderScene = () => {
            
            if (renderer.current === null) return
            if (killRequest.current) return;

            if (currentParams.alive.sum().arraySync() === 0 && nextParams.alive.sum().arraySync() === 0) {
                currentParams = compareWeights(currentParams.fitness, currentParams.weights, nextParams.fitness, nextParams.weights);
                nextParams = generateWeights(currentParams.weights, 0.9, 0.8);
                console.log("Reset")
            };

            requestAnimationFrame(renderScene);
            console.time("renderScene");
            currentParams = forwardBatch(currentParams, targets);
            nextParams = forwardBatch(nextParams, targets);
            renderer.current.updateParams(batchArraySync(currentParams));
            renderer.current.updateParams(batchArraySync(nextParams), B);
            
            controls.update();
            renderer.current.render();
            console.timeEnd("renderScene");
        }


        renderScene();
    });

    const end = () => {
        if (isStarted) {
            killRequest.current = true;
            setIsStarted(false);
        }
    }

    return (
        <ThemeProvider
            attribute="class"
            defaultTheme="dark"
            enableSystem
            disableTransitionOnChange
        >
            
        {
            isTfReady ?
                <>
                    <Menubar className='absolute m-2'>
                        <MenubarMenu>
                            <MenubarTrigger>Menu</MenubarTrigger>
                            <MenubarContent>
                                <MenubarItem onClick={!isStarted ? () => start() : () => end()}>
                                    {!isStarted ? "Start Training" : "End Training"} <MenubarShortcut>Ctrl+S</MenubarShortcut>
                                </MenubarItem>
                                <MenubarItem>Reset Training<MenubarShortcut>Ctrl+R</MenubarShortcut></MenubarItem>
                                <MenubarSeparator />
                                <MenubarItem>Load Checkpoint</MenubarItem>
                                <MenubarSeparator />
                                <MenubarItem>Settings</MenubarItem>
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