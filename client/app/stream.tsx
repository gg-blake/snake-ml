'use client';

import React, { useEffect, useState, useRef, LegacyRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Pane } from "tweakpane";

import "@/app/globals.css";
import { resolve } from 'path';
import { use } from "react";
import { Suspense } from "react";
import { Blade, BladeState } from '@tweakpane/core';

function loadMesh(mesh: THREE.Mesh<THREE.BoxGeometry, THREE.MeshBasicMaterial, THREE.Object3DEventMap>, position: number[]) {
    mesh.position.set(position[0], position[1], position.length > 2 ? position[2] : 0)
}

interface StreamResponse {
  client_id: number;
  snake_data: SnakeData[];
  food_data: number[][];
}

interface SnakeData {
  uid: number;
  pos: number[];
  vel: { [key: number]: number[] };
  score: number;
  alive: boolean;
  history: number[][];
  fitness: number;
}

declare global {
  interface Window { data: StreamResponse | null; }
}

interface Presets extends BladeState {
  
  n_dims: number; // int
  ttl: number; // int
  width: number; // int
  turn_angle: number; // float
  speed: number; // float
  n_snakes: number; // int
  sigma: number; // float
  alpha: number; // float
  hidden_layer_size: number; // int
}

const defaultPreset: Presets = {
  n_dims: 4,
  ttl: 200,
  width: 20,
  turn_angle: -1,
  speed: 1,
  n_snakes: 20,
  sigma: 0.001,
  alpha: 0.00001,
  hidden_layer_size: 20
}

interface ViewPresets extends BladeState {
  colorMode: string;
  viewBounds: boolean;
}

const defaultViewPreset: ViewPresets = {
  colorMode: "alive",
  viewBounds: false,
}


function TweakPaneMenu({ params, viewParams, setParams, setViewParams }: { params: Presets, viewParams: ViewPresets, setParams: React.Dispatch<React.SetStateAction<Presets>>, setViewParams: React.Dispatch<React.SetStateAction<ViewPresets>>}) {
  const paneRef = useRef<Pane | null>(null);

  useEffect(() => {
    if (!paneRef.current) {
      paneRef.current = new Pane();
    }

    // Clear all existing inputs
    paneRef.current.children.forEach(child => paneRef.current!.remove(child));

    const tab = paneRef.current.addTab({
      pages: [
        { title: "Training"},
        { title: "Visual" }
      ]
    })

    tab.pages[0].addBinding(params, 'n_dims', {
      options: {
        "2D": 2,
        "3D": 3,
        "4D": 4
      }
    }).on("change", (e) => {
      setParams(prevParams => ({
        ...prevParams,
        "n_dims": e.value
      }))
    })

    tab.pages[0].addBinding(params, 'ttl', {
      step: 25,
      min: 100,
      max: 1000
    }).on("change", (e) => {
      setParams(prevParams => ({
        ...prevParams,
        "ttl": e.value
      }))
    })

    tab.pages[0].addBinding(params, 'width', {
      step: 1,
      min: 10,
      max: 100
    }).on("change", (e) => {
      setParams(prevParams => ({
        ...prevParams,
        "width": e.value
      }))
    })

    tab.pages[0].addBinding(params, 'turn_angle', {
      step: Math.PI / 16,
      min: Math.PI / 16,
      max: Math.PI / 2
    }).on("change", (e) => {
      setParams(prevParams => ({
        ...prevParams,
        "turn_angle": e.value
      }))
    })

    tab.pages[0].addBinding(params, 'speed', {
      step: 0.01,
      min: 0.05,
      max: 2
    }).on("change", (e) => {
      setParams(prevParams => ({
        ...prevParams,
        "speed": e.value
      }))
    })

    tab.pages[0].addBinding(params, 'n_snakes', {
      step: 1,
      min: 1,
      max: 10000
    }).on("change", (e) => {
      setParams(prevParams => ({
        ...prevParams,
        "n_snakes": e.value
      }))
    })

    tab.pages[0].addBinding(params, 'sigma', {
      step: 0.00001,
      min: 0.00001,
      max: 100
    }).on("change", (e) => {
      setParams(prevParams => ({
        ...prevParams,
        "sigma": e.value
      }))
    })

    tab.pages[0].addBinding(params, 'alpha', {
      step: 0.00001,
      min: 0.00001,
      max: 100
    }).on("change", (e) => {
      setParams(prevParams => ({
        ...prevParams,
        "alpha": e.value
      }))
    })

    tab.pages[0].addBinding(params, 'hidden_layer_size', {
      step: 0.00001,
      min: 0.00001,
      max: 100
    }).on("change", (e) => {
      setParams(prevParams => ({
        ...prevParams,
        "hidden_layer_size": e.value
      }))
    })

    tab.pages[1].addBinding(viewParams, 'colorMode', {
      options: {
        "alive": "alive",
        "best": "best"
      }
    }).on("change", (e) => {
      setViewParams(prevParams => ({
        ...prevParams,
        "colorMode": e.value
      }))
    })

    tab.pages[1].addBinding(viewParams, 'viewBounds').on("change", (e) => {
      setViewParams(prevParams => ({
        ...prevParams,
        "viewBounds": !e.value
      }))
    })

    // Cleanup on component unmount
    return () => {
      paneRef.current?.dispose();
    };
  }, []);

  return <div className='w-auto absolute top left ' ref={(el) => el && paneRef.current && el.appendChild(paneRef.current.element)} />;
}


export default function StreamClient() {
  const mountRef = useRef<HTMLDivElement>(null); // Reference to mount the 3D scene
  const [presets, setPresets] = useState<Presets>(defaultPreset)
  const [viewPresets, setViewPresets] = useState<ViewPresets>(defaultViewPreset)
  const data = useRef<StreamResponse|null>(null);
  
  const [ready, setReady] = useState(false);
  

  
  /*useEffect(() => {
    

    // Refresh the renderer
    if (mountRef.current && mountRef.current.hasChildNodes()) {
      mountRef.current.replaceChildren(); // Clears all previous children
    }
  }, [viewPresets])*/


  

  useEffect(() => {
    console.log(presets)
    const myHeaders = new Headers();
    myHeaders.append("Content-Type", "application/json");
    myHeaders.append('Access-Control-Allow-Origin', 'http://localhost');
    myHeaders.append('Access-Control-Allow-Methods', 'GET, OPTIONS, POST');
    myHeaders.append('Access-Control-Allow-Headers', 'Content-Type');

    const requestOptions = {
      method: "POST",
      headers: myHeaders,
      body: JSON.stringify(presets),
      redirect: "manual"
    };

    // @ts-ignore
    fetch("http://localhost:8000/set", requestOptions)
      .catch((error) => console.error(error));


    // Refresh the renderer
    if (mountRef.current && mountRef.current.hasChildNodes()) {
      mountRef.current.replaceChildren(); // Clears all previous children
    }
  }, [presets])

  const initGame = () => {
    if (data.current == null) {
      return
    }

    let snakePopulation = data.current.snake_data;


    // Set up the camera
    const aspect = window.innerWidth / window.innerHeight;
    const frustumSize = 10; // Controls how "zoomed in" the scene appears

    const camera = new THREE.OrthographicCamera(
      -frustumSize * aspect / 2,  // left
      frustumSize * aspect / 2,   // right
      frustumSize / 2,            // top
      -frustumSize / 2,           // bottom
      0.1,                        // near
      1000                        // far
    );
    camera.position.z = 20;

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

    const light = new THREE.DirectionalLight(0xfffaff, 1);
    light.position.set(5, 5, 5);
    light.castShadow = true; // Allow light to cast shadows
    scene.add(light);

    let uidLookup: {[key: number]: string} = {}
    let uidLookup2: {[key: string]: string} = {}

    const initSnakeMeshes = (group: THREE.Group<THREE.Object3DEventMap>) => {
      for (let i = 0; i < snakePopulation.length; i++) {
        // Load the snake's head's mesh and add it to the snake group
        const currentSnake = snakePopulation[i];
        const snakeMeshHead = new THREE.Mesh(geometry, snakeMaterialHead);
        const currentSnakePosition = currentSnake.pos;
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
    // Load all meshes for the food and add it to the snake group
    let foodGroup = new THREE.Group();

    initSnakeMeshes(snakeGroup);

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

        if (snakePopulation.every((snake: SnakeData) => !snake.alive)) {
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

        snakePopulation.map((snake: SnakeData, snakeIndex: number) => {


          let defaultColor = new THREE.Color("hsl(139, 78%, 63%)");
          

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
