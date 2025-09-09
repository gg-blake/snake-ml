# The Problem
When Tensorflow.js is initialized, it creates a new WebGL context that stores all of its tensors on the GPU as WebGL textures. In many cases of edge visualization, when using Tensorflow.js in tandem with a canvas, it involves a process of downloading the Tensorflow.js tensors from the GPU to the CPU, then uploading a the tensor data back to a new texture on the GPU to be rendered by the canvas. Because of the synchronous nature of this process, it can be a bottleneck for many edge machine learning visualization applications.
# The Solution
We introduce an example solution to unify the WebGL contexts of both a canvas element and the Tensorflow.js backend into one. This enables textures of the tensors to be rendered directly to canvas, skipping the GPU-CPU sync.
# Procedure
1. Initialize the shared WebGL context
```TypeScript
import * as tf from "@tensorflow/tfjs";
const customBackendName = "custom-webgl";

// Get the canvas element from the DOM
// A canvas should automatically have its own WebGL context (if WebGL is supported)
const canvas = document.getElementById('canvas-el');

// Kernels are operations for the GPU (dot product, matrix multiplication, etc.)
// We need to enable each kernel manually for our custom backend
const kernels = tf.getKernelsForBackend("webgl");
kernels.forEach((kernelConfig) => {
	const newKernelConfig = {
		...kernelConfig,
		backendName: customBackendName,
	};
	tf.registerKernel(newKernelConfig);
});

// Initialize the Tensorflow.js WebGL context (referred to as a 'backend') 
const customBackend = new tf.MathBackendWebGL(canvas);
tf.registerBackend(customBackendName, () => customBackend);
await tf.setBackend(customBackendName); // Wait until changes have been made
   ```
2. Initialize our WebGL shader program with a custom vertex and fragment shader
```TypeScript
// Compile vertex shader from glsl source code
const vertexShader = gl.createShader(gl.VERTEX_SHADER)!;
        gl.shaderSource(vertexShader, vertexShaderSource); // Send the source to the shader object
        gl.compileShader(vertexShader); // Compile the shader program

// Compile fragment shader from glsl source code
const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER)!;
gl.shaderSource(fragmentShader, fragmentShaderSource); // Send the source to the shader object
gl.compileShader(fragmentShader); // Compile the shader program

// Create the shader program
const shaderProgram = gl.createProgram();
gl.attachShader(shaderProgram, vertexShader);
gl.attachShader(shaderProgram, fragmentShader);
gl.linkProgram(shaderProgram);
```
`vertexShaderSource`:
```glsl
#version 300 es

in vec4 aVertexPosition;
in vec4 aVertexColor;
in vec3 aVertexNormal;

uniform mat4 uNormalMatrix;
uniform mat4 uModelViewMatrix;
uniform mat4 uProjectionMatrix;
uniform sampler2D uPositionTexture;

out lowp vec4 vColor;
flat out int id;
out highp vec2 coords; // use this if you want to pass the texture sample coords to the fragment shader

void main(void) {
    id = gl_InstanceID; // Get the id of the mesh's instance (built-in)

	// One pixel (RGBA32F) is read per instance from the texture top to bottom, left to right
    ivec2 texShape = textureSize(uPositionTexture, 0);
    float texHeight = float(texShape.y);
    float texWidth = float(texShape.x);
    float v = (floor(float(id) / texWidth) + 0.5) / texHeight;
    float u = (mod(float(id), texWidth) + 0.5) / texWidth;
    coords = vec2(u, v);
    // For this example, the pixel data controls the positioning of a particular cube in 3d space (but can be applied in many ways)
    vec4 offset = texture(uPositionTexture, coords);

    // Transform perspective + offset
    vec4 worldPos = uModelViewMatrix * aVertexPosition;
    gl_Position = uProjectionMatrix * (worldPos + vec4(offset.rgb, 0.0)); // use 0.0 as w for offset

    // Pass color to fragment shader
    vColor = aVertexColor;
}
```
`fragmentShaderSource`:
```glsl
#version 300 es
precision highp float;

in lowp vec4 vColor;
flat in int id;
out lowp vec4 fragColor;
in highp vec2 coords;

void main(void) {
	// In this exmaple, we don't modify the fragment shader based on textures (but you can load another tensor texture to be handled by the fragment shader)
    fragColor = vColor;
}
```
3. In Tensorflow.js, tensors are stored as WebGL textures as an array of pixels. Each pixel on the texture has 4 32-bit float values representing the Red (R), Green (G), Blue (B), Alpha (A) channels. Orders can vary, but every four values of a tensor are mapped to a single RGBA32F pixel. To properly load a tensor's texture in the shared WebGL context, first we load the tensor's texture as normal.
```TypeScript
const tensor = tf.randomUniform([30, 20]); // 2d tensor with arbitrary size
const tensorHeight = tensor.shape[0];
const tensorWidth = Math.floor(tensor.shape[1] / 4);

// Get the tensor's underlying texture
const data = tensor.dataToGPU({
	customTexShape: [tensorHeight, tensorWidth] // manually specify the texture dimensions since Tensorflow.js sometimes is wrong
});
const texture = data.texture!;
const canvasWidth = canvas.width;
const canvasHeight = canvas.height;

const gl = canvas.getContext('webgl'); // WebGL context object
```
4. Tensorflow.js when initializing a tensor has a side effects to buffers, before we draw, we must revert the frame buffer, canvas, and vertex array buffer. Otherwise, meshes that were previously loaded to these buffers will be overwritten.
```TypeScript
// Clean the frame buffer
gl.bindFramebuffer(gl.FRAMEBUFFER, null);

// Reset viewport size (because Tensorflow.js will modify this)
gl.viewport(0, 0, canvasWidth, canvasHeight);
gl.scissor(0, 0, canvasWidth, canvasHeight);

// Some tensor artifacts are previously loaded into vertex buffer
gl.bindVertexArray(null);

// Clear the texture
gl.bindTexture(gl.TEXTURE_2D, null);

// Bind our tensor's texture to the first texture buffer
gl.bindTexture(gl.TEXTURE_2D, texture);

// Ensure texture has the right settings for sampling in vertex shader
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); // prevents  sample interpolation on vertex shader
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST); // prevents sample interpolation on vertex shader
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

// Safely unbind the tensor's texture
gl.bindTexture(gl.TEXTURE_2D, null);
```
5. Prepare our cube example mesh to be drawn
```TypeScript
// Clear canvas
gl.clearColor(0.0, 0.0, 0.0, 1.0);
gl.clearDepth(1.0);
gl.enable(gl.DEPTH_TEST);
gl.depthFunc(gl.LEQUAL);
gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

/* Create Cube Buffers */

// Create a buffer for the cube's vertices' positions.
const positionBuffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
gl.bufferData(
	gl.ARRAY_BUFFER,
	new Float32Array(Cube.POSITIONS),
	gl.STATIC_DRAW,
);

// Cube face colors
const colors: number[][] = ...; // [[1.0, 1.0, 1.0, 1.0], ..., [1.0, 1.0, 1.0, 1.0]]
const colorBuffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
gl.bufferData(
	gl.ARRAY_BUFFER,
	new Float32Array(colors),
	gl.STATIC_DRAW,
);

// Cube indices
const indices: number[] = ...; // [0, 1, 2, 0, ..., 22, 20, 22, 23]
const indexBuffer = gl.createBuffer();
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
gl.bufferData(
	gl.ELEMENT_ARRAY_BUFFER,
	new Uint16Array(indices),
	gl.STATIC_DRAW,
);

/* Point Cube Buffers to Program Attributes */

// Position attribute
const vpLocation = gl.getAttribLocation(shaderProgram, "aVertexPosition");
gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
gl.vertexAttribPointer(vpLocation, 3, gl.FLOAT, false, 0, 0);
gl.enableVertexAttribArray(vpLocation);

// Color attribute
const fcLocation = gl.getAttribLocation(shaderProgram, "aVertexColor");
gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
gl.vertexAttribPointer(fcLocation, 4, gl.FLOAT, false, 0, 0);
gl.enableVertexAttribArray(fcLocation);

// Index buffer
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
```
6. Now that our cube buffers have been loaded into our WebGL shader program, to handle our tensor's texture in our vertex shader code, we need to load the texture as a uniform into the program
```TypeScript
// Position Texture uniform
const ptLocation = gl.getUniformLocation(shaderProgram, "uPositionTexture");
gl.activeTexture(gl.TEXTURE0);
gl.bindTexture(gl.TEXTURE_2D, positionTexture);
gl.uniform1i(ptLocation, 0);
```
7. Load the uniforms for the cube's perspective matrices
```TypeScript
import { mat4 } from "gl-matrix";

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

// Projection Matrix uniform
const pmLocation = gl.getUniformLocation(shaderProgram, "uProjectionMatrix");
gl.uniformMatrix4fv(pmLocation, false, projectionMatrix);

// Model View Matrix uniform
const mvLocation = gl.getUniformLocation(shaderProgram, "uModelViewMatrix");
gl.uniformMatrix4fv(mvLocation, false, modelViewMatrix);
```
8. Lastly, we draw our cubes. WebGL has a feature for instanced drawing so we can draw multiple copies of one element without loading the buffers for each copy. If we want each pixel of the texture to map to an individual cube, we make sure the cube instances is equal to the number of values in the tensor divided by four.
```TypeScript
const vertexCount = 36; // Number of vertices in a cube
const type = gl.UNSIGNED_SHORT;
const offset = 0;
gl.drawElementsInstanced(
	gl.TRIANGLES,
	vertexCount,
	type,
	offset,
	instanceCount,
);
```
*For rendering subsequent changes to the tensor values, you can repeat steps 3 through 8.*
