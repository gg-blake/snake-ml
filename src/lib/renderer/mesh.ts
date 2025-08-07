import Logging from "./logger";

interface MeshBuffers {
    position: WebGLBuffer;
    color: WebGLBuffer;
    index: WebGLBuffer;
    normal: WebGLBuffer;
}

interface Mesh extends Logging {
    _gl: WebGL2RenderingContext;
    buffers: MeshBuffers | null;
    
    initBuffers(gl: WebGL2RenderingContext): void;
    _initPositionBuffer(gl: WebGL2RenderingContext): WebGLBuffer;
    _initIndexBuffer(gl: WebGL2RenderingContext): WebGLBuffer;
    _initColorBuffer(gl: WebGL2RenderingContext): WebGLBuffer;
    _initNormalBuffer(gl: WebGL2RenderingContext): WebGLBuffer;
    update(positionTexture: WebGLTexture, colorTexture: WebGLTexture, program: WebGLProgram): void;
}

export class Cube extends Logging implements Mesh {
    _gl: WebGL2RenderingContext;
    buffers: MeshBuffers | null;

    constructor(gl: WebGL2RenderingContext) {
        super();
        this._gl = gl;
        this.buffers = null;
    }
    
    initBuffers() {
        // ---Init Position Buffer---
        const position = this._initPositionBuffer();
        // ---Init Index Buffer---
        const index = this._initIndexBuffer();
        // ---Init Color Buffer---
        const color = this._initColorBuffer();
        // ---Init Normal Buffer---
        const normal = this._initNormalBuffer();
        this.buffers = {
            position,
            index,
            color,
            normal,
        };
    }

    _initPositionBuffer(): WebGLBuffer {
        // Create a buffer for the square's positions.
        const positionBuffer = this._gl.createBuffer();

        // Select the positionBuffer as the one to apply buffer
        // operations to from here out.
        this._gl.bindBuffer(this._gl.ARRAY_BUFFER, positionBuffer);

        // Now pass the list of positions into WebGL to build the
        // shape. We do this by creating a Float32Array from the
        // JavaScript array, then use it to fill the current buffer.
        this._gl.bufferData(
            this._gl.ARRAY_BUFFER,
            new Float32Array(Cube.POSITIONS),
            this._gl.STATIC_DRAW,
        );
        return positionBuffer;
    }

    _initColorBuffer(): WebGLBuffer {
        // Convert the array of colors into a table for all the vertices.
        let colors: number[] = [];
        for (let j = 0; j < Cube.FACE_COLORS.length; ++j) {
            const c = Cube.FACE_COLORS[j]!;
            // Repeat each color four times for the four vertices of the face
            colors = colors.concat(c, c, c, c);
        }

        const colorBuffer = this._gl.createBuffer();
        this._gl.bindBuffer(this._gl.ARRAY_BUFFER, colorBuffer);
        this._gl.bufferData(
            this._gl.ARRAY_BUFFER,
            new Float32Array(colors),
            this._gl.STATIC_DRAW,
        );
        return colorBuffer;
    }

    _initIndexBuffer(): WebGLBuffer {
        const indexBuffer = this._gl.createBuffer();
        this._gl.bindBuffer(this._gl.ELEMENT_ARRAY_BUFFER, indexBuffer);

        // Now send the element array to GL
        this._gl.bufferData(
            this._gl.ELEMENT_ARRAY_BUFFER,
            new Uint16Array(Cube.INDICES),
            this._gl.STATIC_DRAW,
        );

        return indexBuffer;
    }

    _initNormalBuffer(): WebGLBuffer {
        const normalBuffer = this._gl.createBuffer();
        this._gl.bindBuffer(this._gl.ARRAY_BUFFER, normalBuffer);

        this._gl.bufferData(
            this._gl.ARRAY_BUFFER,
            new Float32Array(Cube.VERTEX_NORMALS),
            this._gl.STATIC_DRAW,
        );

        return normalBuffer;
    }

    update(
        positionTexture: WebGLTexture,
        colorTexture: WebGLTexture,
        program: WebGLProgram
    ) {
        if (!this.buffers) {
            throw new Error("Mesh buffers not initialized");
        }
        const gl = this._gl;

        // Position attribute
        const vpLocation = gl.getAttribLocation(program, "aVertexPosition");
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.position);
        gl.vertexAttribPointer(vpLocation, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(vpLocation);

        // Color attribute
        const fcLocation = gl.getAttribLocation(program, "aVertexColor");
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.color);
        gl.vertexAttribPointer(fcLocation, 4, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(fcLocation);

        // Index buffer
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.buffers.index);

        // Position Texture uniform
        const ptLocation = gl.getUniformLocation(program, "uPositionTexture");
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, positionTexture);
        gl.uniform1i(ptLocation, 0);

        // Color Texture uniform
        const ctLocation = gl.getUniformLocation(program, "uColorTexture");
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, colorTexture);
        gl.uniform1i(ctLocation, 1);
    }
    
    static FACE_COLORS = [
        [1.0, 1.0, 1.0, 1.0], // Front face: white
        [1.0, 0.0, 0.0, 1.0], // Back face: red
        [0.0, 1.0, 0.0, 1.0], // Top face: green
        [0.0, 0.0, 1.0, 1.0], // Bottom face: blue
        [1.0, 1.0, 0.0, 1.0], // Right face: yellow
        [1.0, 0.0, 1.0, 1.0], // Left face: purple
    ];
    
    static INDICES = [
        0,
        1,
        2,
        0,
        2,
        3, // front
        4,
        5,
        6,
        4,
        6,
        7, // back
        8,
        9,
        10,
        8,
        10,
        11, // top
        12,
        13,
        14,
        12,
        14,
        15, // bottom
        16,
        17,
        18,
        16,
        18,
        19, // right
        20,
        21,
        22,
        20,
        22,
        23, // left
    ];
    
    static POSITIONS = [
        // Front face
        -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0,
    
        // Back face
        -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0,
    
        // Top face
        -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0,
    
        // Bottom face
        -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0,
    
        // Right face
        1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0,
    
        // Left face
        -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0,
    ];
    
    static VERTEX_NORMALS = [
        // Front
        0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
    
        // Back
        0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0,
    
        // Top
        0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
    
        // Bottom
        0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0,
    
        // Right
        1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    
        // Left
        -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0,
    ];
}

export type { Mesh, MeshBuffers };