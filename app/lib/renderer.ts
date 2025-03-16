import * as THREE from 'three';
import { settings } from "../settings";
import { Data } from './model';
import * as tf from '@tensorflow/tfjs';

class Renderer {
    scene: THREE.Scene;
    renderer: THREE.WebGLRenderer;
    camera: THREE.Camera;
    uuids: {[key: number]: string[]};
    group: string[][];
    
    // Generic geometry
    static primaryGeometry = new THREE.BoxGeometry();
    // Snake head material (current population)
    static primaryGameObjectMaterialCurrent = new THREE.MeshStandardMaterial({ color: 0x72e66c, roughness: 1 });
    // Snake tail material (current population)
    static secondaryGameObjectMaterialCurrent = new THREE.MeshStandardMaterial({ color: 0x8aace3, roughness: 1 });
    // Snake head material (next population)
    static primaryGameObjectMaterialNext = new THREE.MeshStandardMaterial({ color: 0xe6e275, roughness: 1 });
    // Snake tail material (next population)
    static secondaryGameObjectMaterialNext = new THREE.MeshStandardMaterial({ color: 0xf584ad, roughness: 1 });
    // Snake dead material
    static tertiaryGameObjectMaterial = new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 1 });
    // Food material (most recently generated food)
    static primaryFoodMaterial = new THREE.MeshStandardMaterial({ color: 0xb37e56, roughness: 1 });
    // Food material (older food)
    static secondaryFoodMaterial = new THREE.MeshStandardMaterial({ color: 0x403f23, roughness: 1 });
    
    // Generic material for debugging
    static debugMaterial = new THREE.MeshStandardMaterial({ color: 0x00ff44, roughness: 1 });
    // Generic line material for debugging
    static debugLineMaterial = new THREE.LineBasicMaterial({ color: 0x0000ff });

    constructor(scene: THREE.Scene, renderer: THREE.WebGLRenderer, camera: THREE.Camera) {
        this.scene = scene;
        this.renderer = renderer;
        this.camera = camera;
        this.uuids = {};
        this.group = [];
    }

    // Add a position to the scene
    addMesh(scene: THREE.Scene, position: number[], geometry: THREE.BufferGeometry, material: THREE.MeshStandardMaterial): string {
        // Check if the parameters are valid
        if (position.length < 2) {
            throw new RangeError("Position must be length greater than 2");
        }

        // Add the position to the scene
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(position[0], position[1], position.length > 2 ? position[2] : 0);
        scene.add(mesh);
        return mesh.uuid;
    }

    addLine(scene: THREE.Scene, p0: number[], p1: number[], material: THREE.LineBasicMaterial): string {
        // Check if the parameters are valid
        if (p0.length < 2 || p1.length < 2) {
            throw new RangeError("Position must be length greater than 2");
        }

        // Add the line to the scene
        const points: THREE.Vector3[] = [];
        points.push(new THREE.Vector3(p0[0], p0[1], p0.length > 2 ? p0[2] : 0));
        points.push(new THREE.Vector3(p1[0], p1[1], p1.length > 2 ? p1[2] : 0));
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const line = new THREE.Line(geometry, material);
        scene.add(line);
        return line.uuid;
    }

    // Helper function adds a single population agent to the scene
    addParam(params: Data<number>, index: number, offset: number) {
        // Initialize the storage for THREE.js mesh uuids 
        this.uuids[index] = []

        // Add mesh for the head (special material)
        const firstPosition = params.history[index - offset][0];
        const firstUUID = this.addMesh(this.scene, firstPosition, Renderer.primaryGeometry, index < settings.model.B ? Renderer.primaryGameObjectMaterialCurrent : Renderer.primaryGameObjectMaterialNext);
        this.uuids[index].push(firstUUID);

        // Add rest of the body (default material)
        for (let historyIndex = 1; historyIndex < params.target[index - offset] + settings.model.startingLength; historyIndex++) {
            const position: number[] = params.history[index - offset][historyIndex];
            const uuid = this.addMesh(this.scene, position, Renderer.primaryGeometry, index < settings.model.B ? Renderer.secondaryGameObjectMaterialCurrent : Renderer.secondaryGameObjectMaterialNext);
            this.uuids[index].push(uuid);
        }
    }

    // Add a population to the scene
    addBatchParams(params: Data<number>, offset: number = 0) {
        for (let paramIndex = offset; paramIndex < settings.model.B + offset; paramIndex++) {
            this.addParam(params, paramIndex, offset);
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

    updateLinePosition(scene: THREE.Scene, uuid: string, p0: number[], p1: number[]) {
        const line = scene.getObjectByProperty('uuid', uuid) as THREE.Line;
        const points: THREE.Vector3[] = [];
        points.push(new THREE.Vector3(p0[0], p0[1], p0.length > 2 ? p0[2] : 0));
        points.push(new THREE.Vector3(p1[0], p1[1], p1.length > 2 ? p1[2] : 0));
        line.geometry.setFromPoints(points);
    }

    // Helper function updates a single population agent in the scene
    updateParam(params: Data<number>, index: number, offset: number) {
        // Only update the meshes if gameObject is still alive
        /*if (!gameObject.alive) {
            return;
        }*/

        // Update all the existing meshes' positions
        for (let uuidsIndex = 0; uuidsIndex < this.uuids[index].length - 1; uuidsIndex++) {
            const position = params.history[index - offset][uuidsIndex];
            const uuid = this.uuids[index][uuidsIndex];
            this.updateMeshPosition(this.scene, uuid, position, 0.7);
        }

        // Add new meshes to the scene that have been added to the indexed agent's history
        for (let historyIndex = this.uuids[index].length; historyIndex < params.target[index - offset] + settings.model.startingLength; historyIndex++) {
            const position = params.history[index - offset][historyIndex];
            const uuid = this.addMesh(this.scene, position, Renderer.primaryGeometry, index < settings.model.B ? Renderer.secondaryGameObjectMaterialCurrent : Renderer.secondaryGameObjectMaterialNext);
            this.uuids[index].push(uuid);
        }

        // Update the material type of the leading mesh in the indexed agent's history
        const lastUUID = this.uuids[index][params.target[index - offset] + settings.model.startingLength - 1];
        this.updateMeshMaterial(this.scene, lastUUID, index < settings.model.B ? Renderer.primaryGameObjectMaterialCurrent : Renderer.primaryGameObjectMaterialNext);
    }

    // Update a population in the scene
    updateParams(params: Data<number>, offset: number = 0) {
        for (let paramIndex = offset; paramIndex < settings.model.B + offset; paramIndex++) {
            this.updateParam(params, paramIndex, offset);
        }
    }

    addGroup(positions: number[][], material: THREE.MeshStandardMaterial) {
        this.group.push([]);
        for (let groupIndex = 0; groupIndex < positions.length; groupIndex++) {
            const currentPosition = positions[groupIndex];
            const uuid = this.addMesh(this.scene, currentPosition, Renderer.primaryGeometry, material);
            this.group[this.group.length - 1].push(uuid);
        }
    }

    addLineGroup(p0: number[][], p1: number[][]) {
        if (p0.length != p1.length) {
            throw new Error("List of start points and end points must be 1 to 1");
        }

        this.group.push([]);
        for (let groupIndex = 0; groupIndex < p0.length; groupIndex++) {
            const currentP0 = p0[groupIndex];
            const currentP1 = p1[groupIndex];
            const uuid = this.addLine(this.scene, currentP0, currentP1, Renderer.debugLineMaterial);
            this.group[this.group.length - 1].push(uuid);
        }
    }

    updateGroupPosition(index: number, positions: number[][], lerpAlpha?: number) {
        if (this.group[index].length != positions.length) {
            throw new Error("Number of positions does not match the number of meshes in the indexed group");
        }

        for (let groupIndex = 0; groupIndex < positions.length; groupIndex++) {
            const currentUUID = this.group[index][groupIndex];
            // Update the position
            const currentPosition = positions[groupIndex];
            this.updateMeshPosition(this.scene, currentUUID, currentPosition, lerpAlpha);
            
        }
    }

    updateGroupMaterial(index: number, materials: THREE.MeshStandardMaterial[]) {
        if (this.group[index].length != materials.length) {
            throw new Error("Number of materials does not match the number of meshes in the indexed group");
        }

        for (let groupIndex = 0; groupIndex < materials.length; groupIndex++) {
            const currentUUID = this.group[index][groupIndex];
            // Update the position
            const currentMaterial = materials[groupIndex];
            this.updateMeshMaterial(this.scene, currentUUID, currentMaterial);

        }
    }

    updateLineGroupPosition(index: number, p0: number[][], p1: number[][]) {
        if (this.group[index].length != p0.length || this.group[index].length != p1.length) {
            throw new Error("Number of positions does not match the number of meshes in the indexed group");
        }

        if (p0.length != p1.length) {
            throw new Error("List of start points and end points must be 1 to 1");
        }

        for (let groupIndex = 0; groupIndex < p0.length; groupIndex++) {
            // Update the position
            const currentUUID = this.group[index][groupIndex];
            const currentP0 = p0[groupIndex];
            const currentP1 = p1[groupIndex];
            this.updateLinePosition(this.scene, currentUUID, currentP0, currentP1);
        }
    }

    render() {
        this.renderer.render(this.scene, this.camera);
    }

    // Remove all objects from the scene
    resetScene() {
        this.uuids = {};
        this.scene.clear();
    }
}

export default Renderer;