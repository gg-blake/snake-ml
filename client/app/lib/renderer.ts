import * as THREE from 'three';
import { B, STARTING_SCORE } from "./constants";
import { BatchParamsArray } from './model';

class Renderer {
    scene: THREE.Scene;
    renderer: THREE.WebGLRenderer;
    camera: THREE.Camera;
    uuids: {[key: number]: string[]};
    
    // Generic geometry
    static readonly primaryGeometry = new THREE.BoxGeometry();
    // Snake head material (current population)
    static readonly primaryGameObjectMaterialCurrent = new THREE.MeshStandardMaterial({ color: 0x72e66c, roughness: 1 });
    // Snake tail material (current population)
    static readonly secondaryGameObjectMaterialCurrent = new THREE.MeshStandardMaterial({ color: 0x8aace3, roughness: 1 });
    // Snake head material (next population)
    static readonly primaryGameObjectMaterialNext = new THREE.MeshStandardMaterial({ color: 0xe6e275, roughness: 1 });
    // Snake tail material (next population)
    static readonly secondaryGameObjectMaterialNext = new THREE.MeshStandardMaterial({ color: 0xf584ad, roughness: 1 });
    // Snake dead material
    static readonly tertiaryGameObjectMaterial = new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 1 });
    // Food material (most recently generated food)
    static readonly primaryFoodMaterial = new THREE.MeshStandardMaterial({ color: 0xb37e56, roughness: 1 });
    // Food material (older food)
    static readonly secondaryFoodMaterial = new THREE.MeshStandardMaterial({ color: 0x403f23, roughness: 1 });

    constructor(scene: THREE.Scene, renderer: THREE.WebGLRenderer, camera: THREE.Camera) {
        this.scene = scene;
        this.renderer = renderer;
        this.camera = camera;
        this.uuids = {}
    }

    // Add a position to the scene
    addMesh(scene: THREE.Scene, position: number[], material: THREE.MeshStandardMaterial): string {
        // Check if the parameters are valid
        if (position.length < 2) {
            throw new RangeError("Position must be length greater than 2");
        }

        // Add the position to the scene
        const mesh = new THREE.Mesh(Renderer.primaryGeometry, material);
        mesh.position.set(position[0], position[1], position.length > 2 ? position[2] : 0);
        scene.add(mesh);
        return mesh.uuid
    }

    // Helper function adds a single population agent to the scene
    addParam(params: BatchParamsArray, index: number, offset: number) {
        // Initialize the storage for THREE.js mesh uuids 
        this.uuids[index] = []

        // Add mesh for the head (special material)
        const firstPosition = params.history[index - offset][0];
        const firstUUID = this.addMesh(this.scene, firstPosition, index < B ? Renderer.primaryGameObjectMaterialCurrent : Renderer.primaryGameObjectMaterialNext);
        this.uuids[index].push(firstUUID);

        // Add rest of the body (default material)
        for (let historyIndex = 1; historyIndex < params.targetIndices[index - offset] + STARTING_SCORE; historyIndex++) {
            const position: number[] = params.history[index - offset][historyIndex];
            const uuid = this.addMesh(this.scene, position, index < B ? Renderer.secondaryGameObjectMaterialCurrent : Renderer.secondaryGameObjectMaterialNext);
            this.uuids[index].push(uuid);
        }
    }

    // Add a population to the scene
    addBatchParams(params: BatchParamsArray, offset: number = 0) {
        for (let paramIndex = offset; paramIndex < B + offset; paramIndex++) {
            this.addParam(params, paramIndex, offset);
        }
    }

    // Add all the targets to the scene
    addTargets(params: BatchParamsArray, targets: number[][]) {
        // Add most recent target to scene (special material)
        const mostRecentPosition = targets[Math.max(...params.targetIndices)];
        this.addMesh(this.scene, mostRecentPosition, Renderer.primaryFoodMaterial);

        // Add rest of the targets to the scene (default material)
        for (let targetIndex = 0; targetIndex < Math.max(...params.targetIndices); targetIndex++) {
            const position: number[] = targets[targetIndex];
            this.addMesh(this.scene, position, Renderer.secondaryFoodMaterial);
        }
    }

    // Update an existing mesh's position in the scene
    updateMeshPosition(scene: THREE.Scene, uuid: string, position: number[], lerpAlpha: number = 1) {
        const mesh = scene.getObjectByProperty('uuid', uuid);
        if (mesh == undefined) {
            throw new Error("The uuid specified could not be found");
        }
        if (!(mesh instanceof THREE.Mesh)) {
            throw new TypeError("The object with the specified uuid is of the wrong type. Should be of type THREE.Mesh");
        }
        if (position.length < 2) {
            throw new RangeError("position must be length greater than 2");
        }

        const positionVector = new THREE.Vector3(position[0], position[1], position.length > 2 ? position[2] : 0);
        mesh.position.lerp(positionVector, lerpAlpha);
    }

    // Update an existing mesh's material in the scene
    updateMeshMaterial(scene: THREE.Scene, uuid: string, newMaterial: THREE.MeshStandardMaterial) {
        const mesh = scene.getObjectByProperty('uuid', uuid);
        if (mesh == undefined) {
            throw new Error("The uuid specified could not be found");
        }
        if (!(mesh instanceof THREE.Mesh)) {
            throw new TypeError("The object with the specified uuid is of the wrong type. Should be of type THREE.Mesh");
        }
        
        mesh.material = newMaterial;
    }

    // Helper function updates a single population agent in the scene
    updateParam(params: BatchParamsArray, index: number, offset: number) {
        // Only update the meshes if gameObject is still alive
        /*if (!gameObject.alive) {
            return;
        }*/

        // Update all the existing meshes' positions
        for (let uuidsIndex = 0; uuidsIndex < this.uuids[index].length - 1; uuidsIndex++) {
            const position = params.history[index - offset][uuidsIndex];
            const uuid = this.uuids[index][uuidsIndex];
            this.updateMeshPosition(this.scene, uuid, position, 1);
        }

        // Add new meshes to the scene that have been added to the indexed agent's history
        for (let historyIndex = this.uuids[index].length; historyIndex < params.targetIndices[index - offset] + STARTING_SCORE; historyIndex++) {
            const position = params.history[index - offset][historyIndex];
            const uuid = this.addMesh(this.scene, position, index < B ? Renderer.secondaryGameObjectMaterialCurrent : Renderer.secondaryGameObjectMaterialNext);
            this.uuids[index].push(uuid);
        }

        // Update the material type of the leading mesh in the indexed agent's history
        const lastUUID = this.uuids[index][params.targetIndices[index - offset] + STARTING_SCORE - 1];
        this.updateMeshMaterial(this.scene, lastUUID, index < B ? Renderer.primaryGameObjectMaterialCurrent : Renderer.primaryGameObjectMaterialNext);
    }

    // Update a population in the scene
    updateParams(params: BatchParamsArray, offset: number = 0) {
        for (let paramIndex = offset; paramIndex < B + offset; paramIndex++) {
            this.updateParam(params, paramIndex, offset);
        }
    }

    render() {
        this.renderer.render(this.scene, this.camera);
    }

    // Remove all objects from the scene
    resetScene() {
        this.scene.clear();
    }
}

export default Renderer;