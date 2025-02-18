import { NDGameObject } from "./model";
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const geometry = new THREE.BoxGeometry();
const snakeMaterialHead = new THREE.MeshBasicMaterial({ color: 0x72e66c });
const snakeMaterialTail = new THREE.MeshBasicMaterial({ color: 0x8aace3 });
const snakeMaterialDead = new THREE.MeshBasicMaterial({ color: 0x1a1a1a });
const foodMaterial = new THREE.MeshBasicMaterial({ color: 0x403f23 });

export default class Renderer {
    scene: THREE.Scene;
    renderer: THREE.WebGLRenderer;
    camera: THREE.OrthographicCamera;
    light: THREE.DirectionalLight;
    controls: OrbitControls;
    // Generic geometry
    static readonly primaryGeometry = new THREE.BoxGeometry();
    // Snake head material
    static readonly primaryGameObjectMaterial = new THREE.MeshBasicMaterial({ color: 0x72e66c });
    // Snake tail material
    static readonly secondaryGameObjectMaterial = new THREE.MeshBasicMaterial({ color: 0x8aace3 });
    // Snake dead material
    static readonly tertiaryGameObjectMaterial = new THREE.MeshBasicMaterial({ color: 0x1a1a1a });
    // Food material (most recently generated food)
    static readonly primaryFoodMaterial = new THREE.MeshBasicMaterial({ color: 0xb37e56 });
    // Food material (older food)
    static readonly secondaryFoodMaterial = new THREE.MeshBasicMaterial({ color: 0x403f23 });

    constructor(mountElement: HTMLCanvasElement | HTMLDivElement) {
        this.scene = new THREE.Scene();
        this.renderer = new THREE.WebGLRenderer();
        this.camera = new THREE.OrthographicCamera();
        this.light = new THREE.DirectionalLight();
        this.controls = new OrbitControls(this.camera, mountElement);
        this.controls.update();
    }

    // Set the camera's position and frustrum parameters
    initCamera(frustrumParams: number[], cameraPosition: number[]) {
        // Check if the parameters are valid
        if (frustrumParams.length != 6) {
            throw new RangeError("frustumPosition must be length 6");
        }
        if (cameraPosition.length != 3) {
            throw new RangeError("cameraPosition must be length 3");
        }

        // Set the camera's position
        this.camera.position.set(
            cameraPosition[0],
            cameraPosition[1],
            cameraPosition[2]
        );

        // Set the camera's frustrum parameters
        this.camera.left = frustrumParams[0];
        this.camera.right = frustrumParams[1];
        this.camera.top = frustrumParams[2];
        this.camera.bottom = frustrumParams[3];
        this.camera.near = frustrumParams[4];
        this.camera.far = frustrumParams[5];
    }

    // Set the lights properties
    initLight(lightPosition: number[], color: THREE.ColorRepresentation, intensity: number, castShadow: boolean) {
        // Check if the parameters are valid
        if (lightPosition.length != 3) {
            throw new RangeError("lightPosition must be length 3");
        }

        // Set the light's position
        this.light.position.set(
            lightPosition[0],
            lightPosition[1],
            lightPosition[2]
        )

        // Set the light's color
        this.light.color = new THREE.Color(color);
        // Set the light's intensity
        this.light.intensity = intensity;
        // Cast shadow
        this.light.castShadow = castShadow
        // Add light to the scene
        this.scene.add(this.light);
    }

    // Set the controls' position
    initControls(controlsPosition: number[]) {
        // Check if the parameters are valid
        if (controlsPosition.length != 3) {
            throw new RangeError("controlsPosition must be length 3");
        }

        // Set the controls' position
        this.controls.target.set(
            controlsPosition[0],
            controlsPosition[1],
            controlsPosition[2]
        );
    }

    // Add a position to the scene
    addMesh(scene: THREE.Scene, position: number[], material: THREE.MeshBasicMaterial): string {
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

    // Add a game object to the scene
    addGameObject(gameObject: NDGameObject) {
        for (let historyIndex = 0; historyIndex < gameObject.history.length - 1; historyIndex++) {
            const position = gameObject.history[historyIndex].arraySync();
            const uuid = this.addMesh(this.scene, position, Renderer.secondaryGameObjectMaterial);
            gameObject.uuids.push(uuid);
        }

        const lastPosition = gameObject.history[gameObject.history.length - 1].arraySync();
        const lastUUID = this.addMesh(this.scene, lastPosition, Renderer.primaryGameObjectMaterial);
        gameObject.uuids.push(lastUUID);
    }

    // Add food to the scene
    addFoodArray(foodArray: number[][]) {
        for (let foodArrayIndex = 0; foodArrayIndex < foodArray.length - 1; foodArrayIndex++) {
            const position = foodArray[foodArrayIndex];
            this.addMesh(this.scene, position, Renderer.secondaryFoodMaterial);
        }

        const lastPosition = foodArray[foodArray.length - 1];
        this.addMesh(this.scene, lastPosition, Renderer.primaryFoodMaterial);
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
    updateMeshMaterial(scene: THREE.Scene, uuid: string, newMaterial: THREE.MeshBasicMaterial) {
        const mesh = scene.getObjectByProperty('uuid', uuid);
        if (mesh == undefined) {
            throw new Error("The uuid specified could not be found");
        }
        if (!(mesh instanceof THREE.Mesh)) {
            throw new TypeError("The object with the specified uuid is of the wrong type. Should be of type THREE.Mesh");
        }
        
        mesh.material = newMaterial;
    }

    // Update a game object in the scene
    updateGameObject(gameObject: NDGameObject) {
        // Only update the meshes if gameObject is still alive
        if (!gameObject.alive) {
            return;
        }

        // Update all the existing meshes' positions
        for (let uuidsIndex = 0; uuidsIndex < gameObject.uuids.length - 1; uuidsIndex++) {
            const position = gameObject.history[uuidsIndex].arraySync();
            const uuid = gameObject.uuids[uuidsIndex];
            this.updateMeshPosition(this.scene, uuid, position, 1);
        }

        // Add new meshes to the scene that have been added to the game object's history
        for (let historyIndex = gameObject.uuids.length; historyIndex < gameObject.history.length; historyIndex++) {
            const position = gameObject.history[historyIndex].arraySync();
            const uuid = this.addMesh(this.scene, position, Renderer.secondaryGameObjectMaterial);
            gameObject.uuids.push(uuid);
        }

        // Update the material type of the leading mesh in the game object's history
        /*const lastUUID = gameObject.uuids[gameObject.uuids.length - 1];
        this.updateMeshMaterial(this.scene, lastUUID, Renderer.secondaryGameObjectMaterial);*/
    }

    render() {
        this.renderer.render(this.scene, this.camera);
    }

    // Remove all objects from the scene
    resetScene() {
        this.scene.clear();
    }
}