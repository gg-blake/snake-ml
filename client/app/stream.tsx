'use client';

import React, { useEffect, useState, useRef } from 'react';
import * as THREE from 'three';

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

export default function StreamClient() {
  const mountRef = useRef<HTMLDivElement>(null); // Reference to mount the 3D scene

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
    // 1. Create the scene
    let scene = new THREE.Scene();
    const aspect = window.innerWidth / window.innerHeight;
    const frustumSize = 10; // This defines how "zoomed in" the camera is
    
    const camera = new THREE.OrthographicCamera(
      -frustumSize * aspect / 2,  // left
      frustumSize * aspect / 2,   // right
      frustumSize / 2,            // top
      -frustumSize / 2,           // bottom
      0.1,                        // near plane
      100                         // far plane
    );
    camera.position.set(40, 10, 30); // Position it away from the origin
    camera.lookAt(0, 0, 0);
    
    camera.zoom = 0.5; // Default is 1. Lower it to zoom out (try 0.5, 0.25, etc.)
    camera.updateProjectionMatrix(); // This is required after changing zoom
    
    const renderer = new THREE.WebGLRenderer();

    renderer.setSize(window.innerWidth, window.innerHeight);
    mountRef.current?.appendChild(renderer.domElement); // Attach renderer to the DOM
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(40, 10, 30); // Position the light
    scene.add(directionalLight);
    
    console.log(window.data?.snake_data)

    // 2. Create a simple cube
    const geometry = new THREE.BoxGeometry(); // 8aace3
    const snake_head_material = new THREE.MeshBasicMaterial({ color: 0x609af7 }); // Green cube
    const snake_tail_material = new THREE.MeshBasicMaterial({ color: 0x8aace3 }); // Green cube
    const food_material = new THREE.MeshBasicMaterial({ color: 0xfaf561 });

    camera.position.z = 20; // Move camera back so we can see the cube
    console.log(window.data?.snake_data)
    // 3. Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      scene = new THREE.Scene();
      
      // Add the snake to the scene
      const snake_pos = window.data?.snake_data
      snake_pos?.map((snake) => {
        // Add the snake head to the scene
        const snake_head_mesh = new THREE.Mesh(geometry, snake_head_material);
        
        snake_head_mesh.position.x = snake.pos[0];
        snake_head_mesh.position.y = snake.pos[1];
        //snake_head_mesh.position.z = snake.pos[2];
        scene.add(snake_head_mesh);
        // Add the tail to the scene
        snake.history.map((tail_pos: number[]) => {
          const snake_tail_mesh = new THREE.Mesh(geometry, snake_tail_material);
          /*const edgeGeometry = new THREE.EdgesGeometry(geometry);
          const edgeMaterial = new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 2 });
          const edgeLines = new THREE.LineSegments(edgeGeometry, edgeMaterial);*/
          snake_tail_mesh.position.x = tail_pos[0];
          snake_tail_mesh.position.y = tail_pos[1];
          //snake_tail_mesh.position.z = tail_pos[2];
          //snake_head_mesh.add(edgeLines)
          scene.add(snake_tail_mesh);
        })
      })
      
      // Add the food to the scene
      const food_pos = window.data?.food_data;
      food_pos?.map((food: number[]) => {
        
        const food_mesh = new THREE.Mesh(geometry, food_material);
        food_mesh.position.x = food[0];
        food_mesh.position.y = food[1];
        //food_mesh.position.z = food[2];
        scene.add(food_mesh);
      })
      
      // Render the snakes and food
      renderer.render(scene, camera);
    };

    animate(); // Start the animation

    // 4. Cleanup on component unmount
    return () => {
      mountRef.current?.removeChild(renderer.domElement);
    };
  }, []);

  return (
    <div className="w-screen h-screen absolute bg-white z-[1]">
      <div ref={mountRef}></div>
    </div>
  );
}
