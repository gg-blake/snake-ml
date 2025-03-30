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

export function dotProduct<T extends tf.Tensor>(A: T, B: T, axis?: number, keepDims?: boolean): T {
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
    const div1 = dot1.div(tf.where(dot2.greater(epsilon), dot2, tf.ones([X]).mul(epsilon))) as tf.Tensor1D;
    return div1
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