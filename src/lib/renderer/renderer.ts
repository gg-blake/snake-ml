import * as tf from "@tensorflow/tfjs";
import { mat4, vec3 } from "gl-matrix";
import { vertexShaderSource, fragmentShaderSource } from "./shaders";
import Logging from "../logger";
import { Cube, Mesh, MeshBuffers } from "./mesh";
import Camera from "./camera";
import Lighting from "./lighting";
import Scene from "./scene";
import { Mountable } from "../model/utils/types";

const contextOptions: WebGLContextAttributes = {
    // https://www.khronos.org/registry/webgl/specs/latest/1.0/#5.2
    alpha: false,
    antialias: false,
    premultipliedAlpha: false,
    preserveDrawingBuffer: false,
    depth: false,
    stencil: false,
    failIfMajorPerformanceCaveat: false,
    desynchronized: true,
}

interface RendererOptions {
    verbose?: boolean;
    debug?: true;
}



class Renderer extends Logging implements Mountable {
    _gl: WebGL2RenderingContext;
    options: RendererOptions;
    width: number;
    height: number;
    program: WebGLProgram;
    scene: Scene;

    constructor(canvas: HTMLCanvasElement, options: RendererOptions) {
        super(options.debug);
        this.options = options;
        this.width = canvas.width;
        this.height = canvas.height;
        this._gl = canvas.getContext("webgl2", contextOptions)!;
        this.program = this._initProgram(this._gl);
        this.scene = new Scene(this._gl, this.options.debug);
        // Add lighting to the scene
        this.scene.light = new Lighting(
            this._gl,
            [0.85, 0.8, 0.75],
            [1.0, 1.0, 1.0],
            [0.3, 0.3, 0.3],
        );
        // Add camera to the scene
        this.scene.camera = new Camera(
            this._gl,
            ...[,],
            this.options.verbose,
            this.options.debug,
        );
        // Add cube mesh to the scene
        const cube = new Cube(this._gl, this.options.debug);
        this.scene.add(cube);
    }

    _initProgram(gl: WebGL2RenderingContext) {
        const vertexShader = this._loadShader(
            gl,
            gl.VERTEX_SHADER,
            vertexShaderSource,
        );
        const fragmentShader = this._loadShader(
            gl,
            gl.FRAGMENT_SHADER,
            fragmentShaderSource,
        );

        // Create the shader program
        const shaderProgram = gl.createProgram();
        gl.attachShader(shaderProgram, vertexShader);
        gl.attachShader(shaderProgram, fragmentShader);
        gl.linkProgram(shaderProgram);
        return shaderProgram;
    }

    _loadShader(gl: WebGL2RenderingContext, type: GLenum, source: string) {
        const shader = gl.createShader(type)!;
        gl.shaderSource(shader, source); // Send the source to the shader object
        gl.compileShader(shader); // Compile the shader program
        return shader;
    }

    render(
        positionTexture: WebGLTexture,
        colorTexture: WebGLTexture,
        instanceCount: number,
    ) {
        // Setup base GL state
        this.scene.clear();

        // Set program
        this._gl.useProgram(this.program);

        // Update scene
        this.scene.update([[positionTexture, colorTexture]], this.program);

        // Draw the scene
        this.scene.draw(instanceCount);
    }

    unmount(verbose?: boolean) {
        this.scene.unmount(verbose); // Unmount scene
        const canvas = this._gl.canvas as HTMLCanvasElement;
        /*
        // Removing the canvas from the DOM manually may interfere with benchmarking
        const parent = canvas.parentNode;
        parent?.removeChild(this._gl.canvas as HTMLCanvasElement); // Unmount canvas
        */
        if (verbose) this.log("canvas unmounted");
        delete tf.engine().registryFactory['custom-webgl'];
        delete tf.engine().registry['custom-webgl'];
        tf.setBackend('cpu'); // or any available backend
        if (verbose) this.log("Unregistered shared WebGL context");
        if (verbose) this.log("unmounted");
    }
}

export { Renderer, Cube, Logging, Lighting, Camera, Scene };
export type { MeshBuffers, Mesh };
