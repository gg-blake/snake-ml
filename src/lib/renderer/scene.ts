import Camera from "./camera";
import Lighting from "./lighting";
import Logging from "../logger";
import { Mesh } from "./mesh";

export default class Scene extends Logging {
    _gl: WebGL2RenderingContext;
    meshes: Mesh[];
    light: Lighting | null;
    camera: Camera | null;
    
    constructor(gl: WebGL2RenderingContext, debug?: boolean) {
        super(debug);
        this._gl = gl;
        this.light = null;
        this.camera = null;
        this.meshes = [];
        
        // Setup scene
        // Set clear color to black, fully opaque
        this._gl.clearColor(0.0, 0.0, 0.0, 1.0);
        // Clear the color buffer with specified clear color
        this._gl.clear(this._gl.COLOR_BUFFER_BIT);
    }
    
    add(mesh: Mesh) {
        mesh.initBuffers(this._gl);
        this.meshes.push(mesh);
    }
    
    update(textureMap: [WebGLTexture, WebGLTexture][], program: WebGLProgram) {
        if (!this.camera) {
            throw new Error("Camera not initialized");
        }
        
        if (!this.light) {
            throw new Error("Lighting not initialized");
        }
        
        // Update camera
        const { modelViewMatrix } = this.camera.update(program);
        
        // Update meshes and lig
        for (let i = 0; i < this.meshes.length; i++) {
            this.meshes[i].update(...textureMap[i], program);
        }
        
        // Update lighting for mesh
        for (let i = 0; i < this.meshes.length; i++) {
            this.light.update(program, modelViewMatrix, this.meshes[i].buffers);
        }
        
    }
    
    draw(instanceCount: number) {
        // Draw call
        const vertexCount = 36;
        const type = this._gl.UNSIGNED_SHORT;
        const offset = 0;
        this._gl.drawElementsInstanced(
            this._gl.TRIANGLES,
            vertexCount,
            type,
            offset,
            instanceCount,
        );
    }
    
    // Setup base GL state
    clear() {
        this._gl.clearColor(0.0, 0.0, 0.0, 1.0);
        this._gl.clearDepth(1.0);
        this._gl.enable(this._gl.DEPTH_TEST);
        this._gl.depthFunc(this._gl.LEQUAL);
        this._gl.clear(this._gl.COLOR_BUFFER_BIT | this._gl.DEPTH_BUFFER_BIT);
    }
    
    unmount(verbose?: boolean) {
        if (this.camera) this.camera.unmount(this._gl, verbose);
        if (!verbose) return;
        this.log("unmounted")
    }
}