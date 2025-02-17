'use client';

import React, { useEffect, useState, useRef, LegacyRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Pane } from "tweakpane";

import "@/app/globals.css";
import { resolve } from 'path';
import { use } from "react";
import { Suspense } from "react";
import { Trainer, SnakeModel, NDGameObject, NDVector, BOUNDS_SCALE } from './lib/model';
import { tslFn } from 'three/webgpu';
import * as tf from '@tensorflow/tfjs';


function loadMesh(mesh: THREE.Mesh<THREE.BoxGeometry, THREE.MeshBasicMaterial, THREE.Object3DEventMap>, position: number[]) {
    mesh.position.set(position[0], position[1], position.length > 2 ? position[2] : 0)
}




export default function NativeClient() {
    const mountRef = useRef<HTMLDivElement>(null); // Reference to mount the 3D scen

    const initGame = () => {
        const trainer = new Trainer(3, 10, 0.9, 0.8);
        trainer.loadRenderer(trainer.renderer, mountRef.current!);
        trainer.renderer.setSize(window.innerWidth, window.innerHeight);
        trainer.controls = trainer.loadControls(trainer.renderer.domElement);

        // Render a wireframe bounding box
        const boundBoxGeometry = new THREE.BoxGeometry(BOUNDS_SCALE * 2, BOUNDS_SCALE * 2, trainer.numberOfDimensions > 2 ? BOUNDS_SCALE * 2 : 1); // (width, height, depth)
        const boundBoxWireframe = new THREE.WireframeGeometry(boundBoxGeometry);
        const boundBoxLineSegments = new THREE.LineSegments(boundBoxWireframe, new THREE.LineBasicMaterial({ color: 0xffffff }));
        trainer.scene.add(boundBoxLineSegments);
        
        // Animation loop
        trainer.step();
        /*const animate = () => {
            requestAnimationFrame(animate);
            console.log(trainer)
            
        }

        animate();*/
        
    }

    useEffect(() => {
        initGame();
    }, [])

    return (
        <div className="w-screen h-screen absolute bg-black z-[1]">
            <div className='z-0 absolute' ref={mountRef}></div>
        </div>
    );
}
