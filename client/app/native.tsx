'use client';

import React, { useEffect, useState, useRef, LegacyRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Pane } from "tweakpane";
import * as tf from '@tensorflow/tfjs';
import "@/app/globals.css";
import { resolve } from 'path';
import { use } from "react";
import { Suspense } from "react";
import { Trainer, SnakeModel, NDGameObject, NDVector, BOUNDS_SCALE } from './lib/model';
import Renderer from './lib/renderer';
import { forwardBatch, main, unitTest } from './lib/modelV2';
import '@tensorflow/tfjs-backend-webgl';


function loadMesh(mesh: THREE.Mesh<THREE.BoxGeometry, THREE.MeshBasicMaterial, THREE.Object3DEventMap>, position: number[]) {
    mesh.position.set(position[0], position[1], position.length > 2 ? position[2] : 0)
}




export default function NativeClient() {
    const mountRef = useRef<HTMLDivElement>(null); // Reference to mount the 3D scen

    const initGame = () => {
        const mountElement = mountRef.current;
        /*if (!mountElement) {
            throw new Error("Mount element not found");
        }
        const renderer = new Renderer(mountElement);

        // Set up the camera
        const aspect = window.innerWidth / window.innerHeight;
        const frustumSize = 100; // Controls how "zoomed in" the scene appears
        const cameraFrustrumParams = [
            -frustumSize * aspect / 2,  // left
            frustumSize * aspect / 2,   // right
            frustumSize / 2,            // top
            -frustumSize / 2,           // bottom
            0,                        // near
            1000                        // far
        ]
        renderer.initCamera(cameraFrustrumParams, [500, 500, 500]);
        renderer.initLight([30, 3, 3], 0xfffaff, 20, true);
        renderer.initControls([0, 0, 0]);
        const trainer = new Trainer(3, 10, 0.9, 0.8);
        
        // Animation loop
        trainer.setupCandidates(renderer);
        const animate = () => {
            if (trainer.status == 'active') {
                requestAnimationFrame(animate);
            }
            

            
            
            trainer.evaluatePairsAnimationFrame(renderer);
            
            
            
            renderer.render();
            
        }

        animate();
        trainer.evaluateCandidatesAnimationEnd();*/
        
    }

    const handleClick = () => {
        console.log("Loading model...");
        unitTest();
    }

    useEffect(() => {
        
        //console.log(out);
        //initGame();
    }, [])

    return (
        <div className="w-screen h-screen absolute bg-black z-[1]">
            <button onClick={handleClick}>Click me</button>
        </div>
    );
}
