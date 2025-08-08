import * as tf from '@tensorflow/tfjs';

export function getSubsetsOfSizeK<T>(arr: T[], k: number = 4): T[][] {
    const result: T[][] = [];

    function backtrack(start: number, subset: T[]) {
        // If the subset has reached size k, add it to the result
        if (subset.length === k) {
            result.push([...subset]);
            return;
        }

        for (let i = start; i < arr.length; i++) {
            // Include arr[i] in the subset
            subset.push(arr[i]);

            // Recursively generate subsets including this element
            backtrack(i + 1, subset);

            // Backtrack to remove the last element and try next option
            subset.pop();
        }
    }

    backtrack(0, []);
    return result;
}

export function dotProduct<T extends tf.Tensor>(A: T, B: T, axis?: number, keepDims?: boolean): tf.Tensor {
    if ((axis || 0) >= A.rank || (axis || 0) >= B.rank) {
        throw new Error("Dimension must be less than rank of tensor");
    }

    return tf.tidy(() => tf.mul(A, B).sum(axis, keepDims));
}

// Projects U onto V (with a single batch dimension)
export const project = (U: tf.Tensor2D, V: tf.Tensor2D): tf.Tensor2D => tf.tidy(() => {
    if (U.rank != 2 || V.rank != 2) {
        throw new Error("Rank of U and V must both be 2");
    }
    if (U.shape[0] != V.shape[0] || U.shape[1] != V.shape[1]) {
        throw new Error(`U [Shape: ${U.shape}] and V [Shape: ${V.shape}] must have matching shapes`);
    }

    const X = U.shape[0];
    const Y = U.shape[1];
    // U: (X, Y)
    // V: (X, Y)

    const dot1 = dotProduct<tf.Tensor2D>(U, V, -1); // (X,)
    const dot2 = dotProduct<tf.Tensor2D>(V, V, -1); // (X,)
    const div1 = tf.div(dot1, dot2) // (X,)
    const tile1 = div1.expandDims(-1).tile([1, Y]); // (X, Y)
    const mul1 = tf.mul(tile1, V); // (X, Y)
    return mul1 as tf.Tensor2D
});

export const cosineSimilarity = (U: tf.Tensor2D, V: tf.Tensor2D, epsilon: number = 1e-8): tf.Tensor1D => tf.tidy(() => {
    if (U.rank != 2 || V.rank != 2) {
        throw new Error("Rank of U and V must both be 2");
    }
    if (U.shape[0] != V.shape[0] || U.shape[1] != V.shape[1]) {
        throw new Error(`U [Shape: ${U.shape}] and V [Shape: ${V.shape}] must have matching shapes`);
    }

    const X = U.shape[0];
    const Y = U.shape[1];
    // U: (X, Y)
    // V: (X, Y)
    const dot1 = dotProduct(U, V, -1); // (X,)
    const normU = U.norm(2, -1); // (X,)
    const normV = V.norm(2, -1); // (X,)
    const dot2 = normU.mul(normV); // (X,)
    const div1 = dot1.div(tf.where(dot2.greater(epsilon), dot2, epsilon)) as tf.Tensor1D;
    return div1
})

// Projects row vectors of C to plane composed of row vectors of A and B
// Corresponding row vectors between A and B are assumed to be perpendicular
export const projectToPlane = (A: tf.Tensor2D, B: tf.Tensor2D, C: tf.Tensor2D): tf.Tensor2D => tf.tidy(() => {
    if (A.rank != 2 || B.rank != 2 || C.rank != 2) {
        throw new Error("Rank of U and V must both be 2");
    }
    if (A.shape[0] != B.shape[0] || A.shape[1] != B.shape[1] || A.shape[0] != C.shape[0] || A.shape[1] != C.shape[1]) {
        throw new Error(`A [Shape: ${A.shape}], B [Shape: ${B.shape}], C [Shape: ${C.shape}] must have matching shapes`);
    }

    const projCA = project(C, A);
    const projCB = project(C, B);
    const planeProjCAB = tf.add<tf.Tensor2D>(projCA, projCB);
    return planeProjCAB
})

// Calculates the angle between row vectors of A and projections of row vectors of C onto the AB plane
export const unsignedPlaneAngle = (A: tf.Tensor2D, B: tf.Tensor2D, C: tf.Tensor2D): tf.Tensor1D => tf.tidy(() => {
    if (A.rank != 2 || B.rank != 2 || C.rank != 2) {
        throw new Error("Rank of U and V must both be 2");
    }
    if (A.shape[0] != B.shape[0] || A.shape[1] != B.shape[1] || A.shape[0] != C.shape[0] || A.shape[1] != C.shape[1]) {
        throw new Error(`A [Shape: ${A.shape}], B [Shape: ${B.shape}], C [Shape: ${C.shape}] must have matching shapes`);
    }

    const projC = projectToPlane(A, B, C);
    const dot1 = dotProduct(projC, A, -1);
    const normProjC = projC.norm(2, -1);
    const normA = A.norm(2, -1);
    const mul1 = tf.mul(normProjC, normA);
    const div1 = tf.div(dot1, mul1);
    const theta = tf.acos(div1.clipByValue(-1, 1)) as tf.Tensor1D;
    return theta
})

// Calculates the signed angle (-π, π) between row vectors of A and projections of row vectors of C onto the AB plane
export const signedPlaneAngle = (A: tf.Tensor2D, B: tf.Tensor2D, C: tf.Tensor2D): tf.Tensor1D => tf.tidy(() => {
    if (A.rank != 2 || B.rank != 2 || C.rank != 2) {
        throw new Error("Rank of U and V must both be 2");
    }
    if (A.shape[0] != B.shape[0] || A.shape[1] != B.shape[1] || A.shape[0] != C.shape[0] || A.shape[1] != C.shape[1]) {
        throw new Error(`A [Shape: ${A.shape}], B [Shape: ${B.shape}], C [Shape: ${C.shape}] must have matching shapes`);
    }

    const unsignedThetaPlaneAB = unsignedPlaneAngle(A, B, C);
    const unsignedThetaPlaneBA = tf.sub(Math.PI / 2, unsignedPlaneAngle(B, A, C));
    const sign = unsignedThetaPlaneBA.div(unsignedThetaPlaneBA.abs());
    const theta = unsignedThetaPlaneAB.mul(sign.where(sign.isNaN().logicalNot(), tf.zeros(sign.shape))) as tf.Tensor1D;
    return theta
})

// Length: n - 1
export function generatePlaneIndices(n: number): tf.Tensor2D {
    return tf.tidy(() => {
        return tf.tensor2d(Array.from(Array(n - 1).keys(), (v: number) => [0, v + 1]), ...[,], "int32");
    })
}

export function clamp(x: number, min: number, max: number): number {
    if (x > max) {
        return max
    }

    if (x < min) {
        return min
    }

    return x
}

interface WebGLData {
    gl: WebGL2RenderingContext;
    program: WebGLProgram;
}

// Set up shared WebGL context between canvas and tensorflowjs
async function initContext(canvas: HTMLCanvasElement) {
    const customBackendName = "custom-webgl";

    const kernels = tf.getKernelsForBackend("webgl");
    kernels.forEach((kernelConfig) => {
        const newKernelConfig = {
            ...kernelConfig,
            backendName: customBackendName,
        };
        tf.registerKernel(newKernelConfig);
    });

    const customBackend = new tf.MathBackendWebGL(canvas);
    tf.registerBackend(customBackendName, () => customBackend);
    await tf.setBackend(customBackendName);

    assertSameContext(customBackend, canvas.getContext("webgl2")!);
}

function assertSameContext(
    backend: tf.MathBackendWebGL,
    otherGL: WebGL2RenderingContext,
) {
    if (backend.gpgpu.gl !== otherGL) {
        throw new Error(
            "TensorflowJS backend does not match current canvas WebGL context",
        );
    }
}

function loadTensorData(tensor: tf.Tensor, glData: WebGLData): tf.GPUData {
    const { gl, program } = glData;

    // Get the tensor's underlying texture
    const data = tensor.dataToGPU();
    const texture = data.texture!;
    const canvasWidth = gl.canvas.width;
    const canvasHeight = gl.canvas.height;

    // Cleanup leftover framebuffer from tensorflowjs
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvasWidth, canvasHeight);
    gl.scissor(0, 0, canvasWidth, canvasHeight);
    gl.bindVertexArray(null);

    // Texture settings for the shader program
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindTexture(gl.TEXTURE_2D, texture); // tf.tensor already binds the texture as a side-effect so this is most definitely redundant
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); // prevents  sample interpolation on vertex shader
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST); // prevents sample interpolation on vertex shader
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    // Cleanup
    gl.bindTexture(gl.TEXTURE_2D, null);

    return data;
}



export { loadTensorData, initContext, assertSameContext };