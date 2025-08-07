import Logging from "./logger";
import { mat4 } from "gl-matrix";

export default class Camera extends Logging {
    _gl: WebGL2RenderingContext;
    position: [number, number, number];
    target: [number, number, number];
    up: [number, number, number];
    fov: number;
    aspect: number;
    zNear: number;
    zFar: number;

    constructor(
        gl: WebGL2RenderingContext,
        startingZoom: number = -100,
        verbose?: boolean,
    ) {
        super();
        this._gl = gl;
        this.position = [0, 0, startingZoom]; // eye position
        this.target = [0, 0, 0]; // what you're looking at
        this.up = [0, 1, 0];
        this.fov = (45 * Math.PI) / 180;
        this.aspect = gl.canvas.width / gl.canvas.height;
        this.zNear = 0.1;
        this.zFar = 400.0;
        this.addMouseListener(gl.canvas as HTMLCanvasElement, verbose);
    }

    addMouseListener(canvas: HTMLCanvasElement, verbose?: boolean) {
        let isDragging = false;
        let lastX = 0;
        let lastY = 0;
        let yaw = 0;
        let pitch = 0;
        this.updateCameraDirection(pitch, yaw);

        canvas.addEventListener("mousedown", (e) => {
            isDragging = true;
            lastX = e.clientX;
            lastY = e.clientY;
        });

        canvas.addEventListener("mouseup", () => {
            isDragging = false;
        });

        canvas.addEventListener("mousemove", (e) => {
            if (!isDragging) return;

            const dx = e.clientX - lastX;
            const dy = e.clientY - lastY;
            lastX = e.clientX;
            lastY = e.clientY;

            const sensitivity = 0.005;
            yaw += dx * sensitivity;
            pitch += dy * sensitivity;
            pitch = Math.max(
                -Math.PI / 2 + 0.01,
                Math.min(Math.PI / 2 - 0.01, pitch),
            ); // prevent gimbal lock

            this.updateCameraDirection(pitch, yaw);
        });

        canvas.addEventListener("wheel", (e) => {
            const sensitivity = 0.01;
            this.position[2] += e.deltaY * sensitivity;
            this.updateCameraDirection(pitch, yaw);
            this.log(this.position[2]);
        });

        if (!verbose) return;
        this.log("Mouse event listeners active");
    }

    updateCameraDirection(pitch: number, yaw: number) {
        const x = Math.cos(pitch) * Math.sin(yaw);
        const y = Math.sin(pitch);
        const z = Math.cos(pitch) * Math.cos(yaw);

        const lookDir: [number, number, number] = [x, y, z];
        this.target = [
            this.position[0] + lookDir[0],
            this.position[1] + lookDir[1],
            this.position[2] + lookDir[2],
        ];
    }

    update(program: WebGLProgram) {
        const gl = this._gl;
        const projectionMatrix = mat4.create();
        mat4.perspective(
            projectionMatrix,
            this.fov,
            this.aspect,
            this.zNear,
            this.zFar,
        );
        const modelViewMatrix = mat4.create();
        mat4.translate(modelViewMatrix, modelViewMatrix, [0.0, 0.0, 0.0]);

        // Camera controls
        mat4.targetTo(modelViewMatrix, this.position, this.target, this.up);

        // Projection Matrix uniform
        const pmLocation = gl.getUniformLocation(program, "uProjectionMatrix");
        gl.uniformMatrix4fv(pmLocation, false, projectionMatrix);

        // Model View Matrix uniform
        const mvLocation = gl.getUniformLocation(program, "uModelViewMatrix");
        gl.uniformMatrix4fv(mvLocation, false, modelViewMatrix);

        return { projectionMatrix, modelViewMatrix };
    }

    unmount(gl: WebGL2RenderingContext, verbose?: boolean) {
        const canvas = gl.canvas as HTMLCanvasElement;
        canvas.removeEventListener("mousedown", () => {
            if (!verbose) return;
            this.log("Event Listener (mousedown) removed");
        });
        canvas.removeEventListener("mouseup", () => {
            if (!verbose) return;
            this.log("Event Listener (mouseup) removed");
        });
        canvas.removeEventListener("mousemove", () => {
            if (!verbose) return;
            this.log("Event Listener (mousemove) removed");
        });
        canvas.removeEventListener("wheel", () => {
            if (!verbose) return;
            this.log("Event Listener (wheel) removed");
        });
    }
}