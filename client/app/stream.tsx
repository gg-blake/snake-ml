'use client';

import React, { useEffect, useState, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import Pane from "tweakpane";
import { PresetObject } from 'tweakpane/dist/types/api/preset';
import "@/app/globals.css";
import { isNullOrUndefined } from 'util';

interface StreamResponse {
  client_id: number;
  snake_data: {
    pos: number[];
    vel: {[key: number]: number[]};
    score: number;
    alive: boolean;
    history: number[][];
    fitness: number;
  }[];
  food_data: number[][];
}

declare global {
    interface Window { data: StreamResponse | null; }
}

window.data = window.data || null;

var isLoaded = false;

interface Presets extends PresetObject {
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
  n_dims: 2,
  ttl: 200,
  width: 20,
  turn_angle: 0,
  speed: 1,
  n_snakes: 20,
  sigma: 0.001,
  alpha: 0.00001,
  hidden_layer_size: 20
}

interface ViewPresets extends PresetObject {
  view_fittest: boolean;
  fitness_score_overlay: boolean;
}

const defaultViewPreset: ViewPresets = {
  view_fittest: false,
  fitness_score_overlay: false
}

export default function StreamClient() {
  const mountRef = useRef<HTMLDivElement>(null); // Reference to mount the 3D scene
  const [presets, setPresets] = useState<Presets>(defaultPreset)
  const [viewPresets, setViewPresets] = useState<ViewPresets>(defaultViewPreset)

  useEffect(() => {
    if (isLoaded) {
        return
    }

    isLoaded = true;
    const pane = new Pane({
      title: "Parameters",
      expanded: true,
      container: document.querySelector('.tweakpane-container') as HTMLElement,
    });
    
    const viewPane = new Pane({
      title: "Parameters",
      expanded: true,
      container: document.querySelector('.tweakpane-container-2') as HTMLElement,
    })
    const PARAMS = presets;
    const VIEW_PARAMS = viewPresets;
  
    pane.addInput(PARAMS, "sigma");
    pane.addInput(PARAMS, "alpha");
    pane.addInput(PARAMS, "hidden_layer_size");
    pane.addInput(PARAMS, "ttl");
    pane.addInput(PARAMS, "speed");
    pane.addInput(PARAMS, "n_snakes");
    pane.addInput(PARAMS, "n_dims");
    pane.addInput(PARAMS, "width");
    pane.addInput(PARAMS, "turn_angle");

    viewPane.addInput(VIEW_PARAMS, "view_fittest");
    viewPane.addInput(VIEW_PARAMS, "fitness_score_overlay");

    const exportBtn = pane.addButton({
      title: "Update"
    })

    exportBtn.on('click', () => {
      let state: Presets = pane.exportPreset() as Presets;
      setPresets(state)
    })

    viewPane.on('change', (ev: any) => {
      let state: ViewPresets = viewPane.exportPreset() as ViewPresets;
      setViewPresets(state);
      console.log(state)
    })
  }, [])

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
      .then((response) => response.text())
      .then((result) => console.log(result))
      .catch((error) => console.error(error));
  }, [presets])

  useEffect(() => {
    const eventSource = new EventSource('http://localhost:8000/stream');
    
    eventSource.onmessage = (event) => {
      window.data = JSON.parse(event.data);
    };

    eventSource.onerror = (error) => {
      console.error('EventSource error:', error);
      eventSource.close();
    };

    return () => {
      eventSource.close();
    };
  }, []);
  
  useEffect(() => {
    const createTextSprite = (message: string, parameters: {fontSize: number, borderColor: string, backgroundColor: string} | {} = {}) => {
      const fontSize = parameters.fontSize || 50;
      const borderColor = parameters.borderColor || 'black';
      const backgroundColor = parameters.backgroundColor || 'white';

      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d');
      context!.font = `${fontSize}px Arial`;

      // Get text size and resize canvas
      const textWidth = context!.measureText(message).width;
      canvas.width = textWidth + 20;
      canvas.height = fontSize + 20;

      // Draw background rectangle
      context!.fillStyle = backgroundColor;
      context!.fillRect(0, 0, canvas.width, canvas.height);

      // Draw border
      context!.strokeStyle = borderColor;
      context!.strokeRect(0, 0, canvas.width, canvas.height);

      // Draw text
      context!.font = `${fontSize}px Arial`;
      context!.fillStyle = 'black';
      context!.fillText(message, 10, fontSize + 5);

      // Create texture from canvas
      const texture = new THREE.CanvasTexture(canvas);
      texture.needsUpdate = true;

      const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
      const sprite = new THREE.Sprite(spriteMaterial);
      
      // Scale sprite to a reasonable size in the 3D world
      const spriteScale = 0.02; // Adjust this scale to fit your scene
      sprite.scale.set(canvas.width * spriteScale, canvas.height * spriteScale, 1);
      
      return sprite;
    };
    

    // 1. Create the scene
    let scene = new THREE.Scene();
    const aspect = window.innerWidth / window.innerHeight;
    const frustumSize = 10; // This defines how "zoomed in" the camera is
    
    const camera = new THREE.OrthographicCamera();
    camera.position.set(40, 10, 30); // Position it away from the origin
    camera.lookAt(0, 0, 0);
    
    camera.zoom = 0.1; // Default is 1. Lower it to zoom out (try 0.5, 0.25, etc.)
    camera.updateProjectionMatrix(); // This is required after changing zoom
    
    const renderer = new THREE.WebGLRenderer();

    renderer.setSize(window.innerWidth, window.innerHeight);
    mountRef.current?.appendChild(renderer.domElement); // Attach renderer to the DOM

    const controls = new OrbitControls( camera, renderer.domElement );
    controls.update();

    // 2. Create a simple cube
    const geometry = new THREE.BoxGeometry(); // 8aace3
    const snake_head_material = new THREE.MeshBasicMaterial({ color: 0x72e66c }); // Green cube
    const snake_tail_material = new THREE.MeshBasicMaterial({ color: 0x8aace3 }); // Green cube
    const food_material = new THREE.MeshBasicMaterial({ color: 0xfaf561 });

    const best_snake_material = new THREE.MeshBasicMaterial({color: 0xedb264});
    const irrelevant_snake_material = new THREE.MeshBasicMaterial({color: 0x878787});

    scene = new THREE.Scene();
    //scene.background = new THREE.Color( 0xa0a0a0 );
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(0, 5, 20); // Position the light
    scene.add(directionalLight);

    let snakes = new THREE.Group();
    let food = new THREE.Group();
    
    food.children = [];

    scene.add(snakes)
    scene.add(food)
    let textSprites = new THREE.Group();
    scene.add(textSprites)
    

    camera.position.z = 20; // Move camera back so we can see the cube
    console.log(window.data?.snake_data)
    let delete_queue = [];
    // 3. Animation loop
    
    const animate = () => {
      requestAnimationFrame(animate);
      let best_fitness = 0;
      if (window.data != null) {
        
        const snake_pos = window.data.snake_data
        let cubes_count = 0;
        snake_pos.map((snake, index: number) => {
          cubes_count += snake.history.length
        })
        
        for (let i = 0; i < cubes_count - snakes.children.length; i++) {
          const snake_mesh = new THREE.Mesh(geometry, snake_tail_material);
          snake_mesh.position?.set(0, 0, 0);
          snakes.add(snake_mesh);
        }

        if (textSprites) {
          scene.remove(textSprites);
          textSprites = new THREE.Group();
          scene.add(textSprites);
        }
        
        
        let count_i = 0;
        for (let i = 0; i < snake_pos?.length; i++) {
          if (!snake_pos[i].alive) {
            continue;
          }

          if (viewPresets.fitness_score_overlay) {
            

            const textSprite = createTextSprite(`${snake_pos[i].fitness}`);
            if (textSprites.children[i] == undefined) {
              textSprites.add(textSprite)
              // Position text near the cube (slightly offset on the x-axis)
              textSprite.position.set(snake_pos[i].pos[0] + 1.2, snake_pos[i].pos[1] + 1.2, snake_pos[i].pos[2]);
            } else {
              textSprites.children[i].position.lerp(new THREE.Vector3(snake_pos[i].pos[0] + 1.2, snake_pos[i].pos[1] + 1.2, snake_pos[i].pos[2]), 0.9)
            }
            
            // Make text always face the camera
            textSprite.quaternion.copy(camera.quaternion);
          }

          snake_pos[i].history?.map((pos: number[], index: number) => {
            if (snakes.children[count_i] == undefined) {
              return
            }
            
            if (viewPresets.view_fittest) {
              if (snake_pos[i].fitness >= best_fitness) {
                best_fitness = snake_pos[i].fitness;
                snakes.children[count_i].material = best_snake_material;
              } else {
                snakes.children[count_i].material = irrelevant_snake_material;
              }
            }
              
            if (pos.length == 4) {
              snakes.children[count_i].material = snakes.children[count_i].material.clone()
              snakes.children[count_i].material.color.setHSL((pos[3] + 20) / 40, 1, 0.5);
            }
            
            if (pos.length > 2) {
              snakes.children[count_i].position.lerp(new THREE.Vector3(pos[0], pos[1], pos[2]), 0.5);
            } else {
              snakes.children[count_i].position.set(pos[0], pos[1], 0);
            }
            
            count_i++;
          })
        }

        scene.remove(food);
        food = new THREE.Group();
        scene.add(food);
        window.data.food_data.map((pos: number[], index: number) => {
          const food_mesh = new THREE.Mesh(geometry, food_material);
          if (pos.length < 3) {
            food_mesh.position.set(pos[0], pos[1], 0);
          } else {
            food_mesh.position.set(pos[0], pos[1], pos[2]);
          }
          
          food.add(food_mesh)
        })
        
        //scene.children[1] = null;
        //scene.add(food);

        
       
      }

      //scene.children[0] = snakes;
      //scene.children[1] = food;
      
      
      // Render the snakes and food
      renderer.render(scene, camera);
    };

    animate(); // Start the animation

    // 4. Cleanup on component unmount
    return () => {
      mountRef.current?.removeChild(renderer.domElement);
    };
  }, [presets]);

  return (
    <div className="w-screen h-screen absolute bg-white z-[1]">
      <div className='z-0 absolute' ref={mountRef}></div>
      <div className="absolute tweakpane-container z-10"></div>
      <div className="absolute tweakpane-container-2 z-10 bottom-0"></div>
    </div>
  );
}
