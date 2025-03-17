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

export function generatePlaneIndices(n: number): tf.Tensor2D {
    return tf.tidy(() => {
        const _tensorList: tf.Tensor1D[] = getSubsetsOfSizeK<number>(Array.from(Array(n).keys()), 2)
            .map((v: number[]) => tf.tensor(v, ...[,], 'int32'));
        return tf.stack<tf.Tensor1D>(_tensorList) as tf.Tensor2D;
    })
}