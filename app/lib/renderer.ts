import * as THREE from 'three';
import { settings } from "../settings";
import * as tf from '@tensorflow/tfjs';
import NEAT from './optimizer';
import { calculateNearbyBounds } from './layers/inputnorm';
import { arraySync, Data } from './model';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { clamp } from './util';

const projectDirectionToBounds = (position: tf.Tensor2D, direction: tf.Tensor3D, scale: number): tf.Tensor2D => tf.tidy(() => {
    const B = position.shape[0];
    const C = position.shape[1];
    const distance = calculateNearbyBounds(position, direction, scale).expandDims(-1).tile([1, C]) as tf.Tensor2D;
    
    const forwardVectors = direction.slice([0, 0, 0], [B, 1, C]).squeeze([1]) as tf.Tensor2D;
    const proj = tf.add(position, tf.mul(forwardVectors, distance)) as tf.Tensor2D;
    
    return proj
})

export class Renderer {
    scene: THREE.Scene;
    renderer: THREE.WebGLRenderer;
    camera: THREE.Camera;
    uuids: {[key: number]: string[]};
    group: {[key: string]: string[]};
    
    // Generic geometry
    static primaryGeometry = new THREE.BoxGeometry();
    // Snake head material (current population)
    static primaryMaterial = new THREE.MeshStandardMaterial({ color: 0x72e66c, roughness: 1 });
    // Snake dead material
    static secondaryMaterial = new THREE.MeshStandardMaterial({ color: 0x000000, roughness: 1, opacity: 0 });
    // Food material (most recently generated food)
    static primaryFoodMaterial = new THREE.MeshStandardMaterial({ color: 0xb37e56, roughness: 1 });
    // Food material (older food)
    static secondaryFoodMaterial = new THREE.MeshStandardMaterial({ color: 0x403f23, roughness: 1 });
    
    // Generic material for debugging
    static debugMaterial = new THREE.MeshStandardMaterial({ color: 0x00ff44, roughness: 1 });
    // Generic line material for debugging
    static primaryDebugLineMaterial = new THREE.LineBasicMaterial({ color: 0x5e5e5e });
    // Generic line material for debugging
    static secondaryDebugLineMaterial = new THREE.LineBasicMaterial({ color: 0x000000, opacity: 0.5 });

    constructor(scene: THREE.Scene, renderer: THREE.WebGLRenderer, camera: THREE.Camera) {
        this.scene = scene;
        this.renderer = renderer;
        this.camera = camera;
        this.uuids = {};
        this.group = {};
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
        this.uuids[index] = [];
        const materialClone = Renderer.primaryMaterial.clone();

        // Add rest of the body (default material)
        for (let historyIndex = 0; historyIndex < params.target[index - offset] + settings.model.startingLength; historyIndex++) {
            const position: number[] = params.history[index - offset][historyIndex];
            const uuid = this.addMesh(this.scene, position, Renderer.primaryGeometry, materialClone);
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
    updateMeshMaterial(scene: THREE.Scene, uuid: string, color: THREE.Color, lerpAlpha: number = 1) {
        const mesh = scene.getObjectByProperty('uuid', uuid) as THREE.Mesh;
        if (mesh == undefined) {
            throw new Error("The uuid specified could not be found");
        }
        if (!(mesh instanceof THREE.Mesh)) {
            throw new TypeError("The object with the specified uuid is of the wrong type. Should be of type THREE.Mesh");
        }
        
        (mesh.material as THREE.MeshStandardMaterial).color.lerp(color, lerpAlpha);
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

        const baseMaterial = (this.scene.getObjectByProperty('uuid', this.uuids[index][0]) as THREE.Mesh).material as THREE.MeshStandardMaterial;

        // Update all the existing meshes' positions
        for (let uuidsIndex = 0; uuidsIndex < this.uuids[index].length - 1; uuidsIndex++) {
            const position = params.history[index - offset][uuidsIndex];
            const uuid = this.uuids[index][uuidsIndex];
            this.updateMeshPosition(this.scene, uuid, position, 0.7);
            this.updateMeshMaterial(this.scene, uuid, baseMaterial.color, 0.5);
        }

        // Add new meshes to the scene that have been added to the indexed agent's history
        for (let historyIndex = this.uuids[index].length; historyIndex <= params.target[index - offset] + settings.model.startingLength; historyIndex++) {
            const position = params.history[index - offset][historyIndex];
            const uuid = this.addMesh(this.scene, position, Renderer.primaryGeometry, baseMaterial);
            this.uuids[index].push(uuid);
        }
    }

    // Update a population in the scene
    updateParams(params: Data<number>, offset: number = 0) {
        for (let paramIndex = offset; paramIndex < settings.model.B + offset; paramIndex++) {
            this.updateParam(params, paramIndex, offset);
        }
    }

    // Update a population's materials in the scene
    updateParamsMaterial(params: Data<number>, colors: THREE.Color[]) {
        for (let paramIndex = 0; paramIndex < params.position.length; paramIndex++) {
            const uuid = this.uuids[paramIndex][0];
            this.updateMeshMaterial(this.scene, uuid, colors[paramIndex], 0.7);
            
        }
    }

    addBoxGroup(key: string, positions: number[][], material: THREE.MeshStandardMaterial) {
        if (key in Object.keys(this.group)) {
            throw new Error(`Render group [${key}] already exists`);
        }
        
        this.group[key] = [];
        for (let groupIndex = 0; groupIndex < positions.length; groupIndex++) {
            const currentPosition = positions[groupIndex];
            const uuid = this.addMesh(this.scene, currentPosition, Renderer.primaryGeometry, material);
            this.group[key].push(uuid);
        }
    }

    addLineGroup(key: string, p0: number[][], p1: number[][]) {
        if (p0.length != p1.length) {
            throw new Error("List of start points and end points must be 1 to 1");
        }

        if (key in Object.keys(this.group)) {
            throw new Error(`Render group [${key}] already exists`);
        }

        this.group[key] = [];
        for (let groupIndex = 0; groupIndex < p0.length; groupIndex++) {
            const currentP0 = p0[groupIndex];
            const currentP1 = p1[groupIndex];
            const uuid = this.addLine(this.scene, currentP0, currentP1, Renderer.primaryDebugLineMaterial);
            this.group[key].push(uuid);
        }
    }

    updateGroupPosition(key: string, positions: number[][], lerpAlpha?: number) {
        if (this.group[key].length != positions.length) {
            throw new Error("Number of positions does not match the number of meshes in the indexed group");
        }

        for (let groupIndex = 0; groupIndex < positions.length; groupIndex++) {
            const currentUUID = this.group[key][groupIndex];
            // Update the position
            const currentPosition = positions[groupIndex];
            this.updateMeshPosition(this.scene, currentUUID, currentPosition, lerpAlpha);
            
        }
    }

    updateGroupMaterial(key: string, materials: THREE.MeshStandardMaterial[]) {

        if (this.group[key].length != materials.length) {
            throw new Error("Number of materials does not match the number of meshes in the indexed group");
        }

        for (let groupIndex = 0; groupIndex < materials.length; groupIndex++) {
            const currentUUID = this.group[key][groupIndex];
            // Update the position
            const currentMaterial = materials[groupIndex];
            this.updateMeshMaterial(this.scene, currentUUID, currentMaterial.color);

        }
    }

    updateLineGroupPosition(key: string, p0: number[][], p1: number[][]) {
        if (this.group[key].length != p0.length || this.group[key].length != p1.length) {
            throw new Error("Number of positions does not match the number of meshes in the indexed group");
        }

        if (p0.length != p1.length) {
            throw new Error("List of start points and end points must be 1 to 1");
        }

        for (let groupIndex = 0; groupIndex < p0.length; groupIndex++) {
            // Update the position
            const currentUUID = this.group[key][groupIndex];
            const currentP0 = p0[groupIndex];
            const currentP1 = p1[groupIndex];
            this.updateLinePosition(this.scene, currentUUID, currentP0, currentP1);
        }
    }

    updateLineMaterial(scene: THREE.Scene, uuid: string, material: THREE.LineBasicMaterial) {
        const line = scene.getObjectByProperty('uuid', uuid) as THREE.Line;
        line.material = material;
    }

    updateLineGroupMaterial(key: string, material: THREE.LineBasicMaterial[]) { 
        for (let lineIndex = 0; lineIndex < this.group[key].length; lineIndex++) {
            this.updateLineMaterial(this.scene, this.group[key][lineIndex], material[lineIndex]);
        }
    }

    render() {
        this.renderer.render(this.scene, this.camera);
    }

    // Remove all objects from the scene
    resetScene() {
        this.uuids = {};
        this.group = {};
        this.scene.clear();
    }
}

export class NEATRenderer extends Renderer {
    controls: OrbitControls | null;
    constructor(scene: THREE.Scene, renderer: THREE.WebGLRenderer, camera: THREE.Camera) {
        super(scene, renderer, camera);
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.update();
        this.controls.target.set(0, 0, 0);
    }

    renderInit(model: NEAT) {
        this.addBatchParams(arraySync(model.state));
        this.addBoxGroup('targets', model.target.arraySync(), Renderer.primaryFoodMaterial);
        this.render();
    }

    renderStep(model: NEAT) {
        if (this.controls == null) {
            throw new Error("Must initialize orbit controls first with renderInit()");
        }

        this.updateParams(arraySync(model.state));

        if (settings.renderer.showProjections) {
            this.renderProjections(model);
        } else if (this.group.hasOwnProperty('projections')) {
            this.removeGroup('projections');
        }

        if (settings.renderer.showFitnessDelta) {
            const fitnessDelta = model.fitnessDelta.arraySync() as number[];
            const max = model.fitnessDelta.max().arraySync() as number;
            const min = model.fitnessDelta.min().arraySync() as number;
            const dt = max - min;
            Object.values(this.uuids).map((uuid: string[], index: number) => {
                const mesh = this.scene.getObjectByProperty('uuid', uuid[0]) as THREE.Mesh;
                const delta = fitnessDelta[index];
                (mesh.material as THREE.MeshStandardMaterial).color.setRGB(clamp(1 - (delta - min) / dt, 0, 1), clamp((delta - min) / dt, 0, 1), 0)
            })
            //console.log(colors)
        } else {
            Object.values(this.uuids).map((uuid: string[], index: number) => {
                const mesh = this.scene.getObjectByProperty('uuid', uuid[0]) as THREE.Mesh;
                (mesh.material as THREE.MeshStandardMaterial).color.lerp(Renderer.primaryMaterial.color, 0.5);
            })
        }

        if (settings.renderer.showTargetRays) {
            this.renderTargetRays(model);
        } else if(this.group.hasOwnProperty('target_rays')) {
            this.removeGroup('target_rays');
        }

        if (settings.renderer.showNormals) {
            this.renderNormals(model);
        } else if (this.group.hasOwnProperty('normals')) {
            this.removeGroup('normals');
        }

        tf.tidy(() => {
            const targetIndices = model.state.target.max(0).arraySync() as number;
            this.updateGroupMaterial('targets', Array.from(Array(model.config.T).keys(), (i: number) => i == targetIndices ? NEATRenderer.primaryFoodMaterial : NEATRenderer.secondaryFoodMaterial))
        })

        const alive = model.state.active.arraySync() as number[];
        Object.values(this.uuids).map((uuid: string[], index: number) => {
            if (!alive[index]) {
                const mesh = this.scene.getObjectByProperty('uuid', uuid[0]) as THREE.Mesh;
                (mesh.material as THREE.MeshStandardMaterial).color.setRGB(0, 0, 0);
            }
        })

        if (settings.renderer.showBest) {
            (model.state.fitness.greaterEqual(model.state.fitness.mul(model.state.active.cast('float32')).max()).cast('int32').arraySync() as number[])
            .map((value: number, index: number) => {
                if (value == 0) {
                    const mesh = this.scene.getObjectByProperty('uuid', this.uuids[index][0]) as THREE.Mesh;
                    (mesh.material as THREE.MeshStandardMaterial).color.setRGB(0, 0, 0);
                }
            });
        }

        
        this.controls.update();
        this.render();
    }

    removeGroup(key: string) {
        this.group[key].map((uuid: string) => {
            const mesh = this.scene.getObjectByProperty('uuid', uuid) as THREE.Mesh;
            this.scene.remove(mesh)
        })
        delete this.group[key];
    }

    renderProjections(model: NEAT) {
        tf.tidy(() => {
            const proj = projectDirectionToBounds(model.state.position, model.state.direction, model.config.boundingBoxLength);
            const projectionPositions = (proj.arraySync() as number[][]);
            if (!this.group.hasOwnProperty('projections')) {
                this.addBoxGroup('projections', projectionPositions, Renderer.debugMaterial);
                return;
            }
            this.updateGroupPosition('projections', projectionPositions, 1);
        })
    }

    renderNormals(model: NEAT) {
        tf.tidy(() => {
            const positionTile = model.state.position.expandDims(1).tile([1, model.config.C, 1]).reshape([model.config.B * model.config.C, model.config.C]);
            const directionTile = positionTile.add(model.state.direction.reshape([model.config.B * model.config.C, model.config.C]));
            if (!this.group.hasOwnProperty('normals')) {
                this.addLineGroup('normals', positionTile.arraySync() as number[][], directionTile.arraySync() as number[][]);
                return;
            }
            this.updateLineGroupPosition('normals', positionTile.arraySync() as number[][], directionTile.arraySync() as number[][]);
            
        })
    }

    renderTargetRays(model: NEAT) {
        tf.tidy(() => {
            const targetPositions = model.target.gather(model.state.target).arraySync() as number[][];
            const position = model.state.position.arraySync() as number[][];
            if (!this.group.hasOwnProperty('target_rays')) {
                this.addLineGroup('target_rays', targetPositions, position);
                return;
            }
            this.updateLineGroupPosition('target_rays', targetPositions, position);

            if (settings.renderer.showBest) {
                this.updateLineGroupMaterial('target_rays', (model.state.fitness.greaterEqual(model.state.fitness.mul(model.state.active.cast("float32")).max()).cast('int32').arraySync() as number[]).map((value: number) => value == 1 ? Renderer.primaryDebugLineMaterial : Renderer.secondaryDebugLineMaterial));
                return;
            }

            this.updateLineGroupMaterial('target_rays', (model.state.active.arraySync() as number[]).map((value: number) => value == 1 ? Renderer.primaryDebugLineMaterial : Renderer.secondaryDebugLineMaterial));
        })
    }
}