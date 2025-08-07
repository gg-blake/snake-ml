import { vec3, mat4 } from "gl-matrix";
import Logging from "./logger";
import { MeshBuffers } from "./renderer";

export default class Lighting extends Logging {
    _gl: WebGL2RenderingContext;
    directionalLightPosition: vec3;
    directionLightColor: vec3;
    ambientLightColor: vec3;

    constructor(
        gl: WebGL2RenderingContext,
        directionalLightPosition: [number, number, number],
        directionLightColor: [number, number, number],
        ambientLightColor: [number, number, number],
    ) {
        super();
        this._gl = gl;
        this.directionalLightPosition = vec3.create();
        vec3.copy(this.directionalLightPosition, directionalLightPosition);
        this.directionLightColor = vec3.create();
        vec3.copy(this.directionLightColor, directionLightColor);
        this.ambientLightColor = vec3.create();
        vec3.copy(this.ambientLightColor, ambientLightColor);
    }

    update(program: WebGLProgram, modelViewMatrix: mat4, buffers: MeshBuffers | null) {
        if (!buffers) {
            throw new Error("Mesh buffers not initialized");
        }
        const gl = this._gl;

        // Create normal matrix
        const normalMatrix = mat4.create();
        mat4.invert(normalMatrix, modelViewMatrix);
        mat4.transpose(normalMatrix, normalMatrix);

        // Normal attribute
        const nLocation = gl.getAttribLocation(program, "aVertexNormal");
        gl.bindBuffer(gl.ARRAY_BUFFER, buffers.normal);
        gl.vertexAttribPointer(nLocation, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(nLocation);

        // Normal Matrix uniform
        const nmLocation = gl.getUniformLocation(program, "uNormalMatrix");
        gl.uniformMatrix4fv(nmLocation, false, normalMatrix);

        // Directional Light Position uniform
        const dlLocation = gl.getUniformLocation(
            program,
            "uDirectionalPosition",
        );
        gl.uniform3fv(dlLocation, this.directionalLightPosition);

        // Directional Light Color uniform
        const dlcLocation = gl.getUniformLocation(
            program,
            "uDirectionalIntensity",
        );
        gl.uniform3fv(dlcLocation, this.directionLightColor);

        // Ambient Light Color uniform
        const alLocation = gl.getUniformLocation(program, "uAmbientIntensity");
        gl.uniform3fv(alLocation, this.ambientLightColor);
    }
}