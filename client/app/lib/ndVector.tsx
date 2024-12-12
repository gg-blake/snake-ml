import * as ort from 'onnxruntime';

async function aguileraPerezRotation(v: ort.Tensor, basis: ort.Tensor, angle: number): Promise<ort.Tensor> {
    const [Q, _] = await ort.linalg.qr(basis); // QR decomposition to orthonormalize basis
    const m = Q.dims[1]; // Dimension of the subspace
    const n = Q.dims[0]; // Dimension of the full space

    const vProjected = await ort.matMul(await ort.transpose(Q), v); // Project v into the subspace

    const cosTheta = Math.cos(angle);
    const sinTheta = Math.sin(angle);
    let rotationMatrix = ort.eye(m); // Identity matrix for the subspace
    if (m >= 2) {
        rotationMatrix = ort.eye(m);
        rotationMatrix.set([0, 0], cosTheta);
        rotationMatrix.set([0, 1], -sinTheta);
        rotationMatrix.set([1, 0], sinTheta);
        rotationMatrix.set([1, 1], cosTheta);
    }

    const vRotatedProjected = await ort.matMul(rotationMatrix, vProjected); // Rotate within the subspace
    const vRotated = await ort.matMul(Q, vRotatedProjected); // Map back to the original space

    return vRotated;
}

class NDVector {
    nDims: number;
    dimIds: number[][];
    planeIds: number[][];
    axisLookup: Record<number, ort.Tensor>;

    constructor(nDims = 2) {
        this.nDims = nDims;
        this.dimIds = Array.from({ length: nDims }, (_, i) => [i, i + nDims]);
        this.planeIds = [];

        for (const [axis1Pos, axis1Neg] of this.dimIds) {
            for (const [axis2Pos, axis2Neg] of this.dimIds) {
                if (axis1Pos === axis2Pos || axis1Neg === axis2Neg) continue;
                this.planeIds.push([axis1Pos, axis2Pos, axis1Neg, axis2Neg]);
            }
        }

        this.axisLookup = Object.fromEntries(this.getAxisTensors().map((v, i) => [i, v]));
    }

    getAxisTensors(): ort.Tensor[] {
        const v = ort.stack(
            Array.from({ length: this.nDims }, (_, i) => ort.oneHot(i % this.nDims, this.nDims))
        ).repeat(2, 1);

        const negIndices = Array.from({ length: this.nDims }, (_, i) => i + this.nDims);
        negIndices.forEach(i => v.set(i, v.get(i).mul(-1)));

        return v;
    }

    turnLeft(dim: number) {
        const tmp = this.axisLookup[this.planeIds[dim][0]];
        for (let i = 0; i < 3; i++) {
            this.axisLookup[this.planeIds[dim][i]] = this.axisLookup[this.planeIds[dim][i + 1]];
        }
        this.axisLookup[this.planeIds[dim][3]] = tmp;
    }

    turnRight(dim: number) {
        const tmp = this.axisLookup[this.planeIds[dim][3]];
        for (let i = 3; i > 0; i--) {
            this.axisLookup[this.planeIds[dim][i]] = this.axisLookup[this.planeIds[dim][i - 1]];
        }
        this.axisLookup[this.planeIds[dim][0]] = tmp;
    }

    async turnLeftAngle(dim: number, theta = Math.PI / 2) {
        const basisVectorX = this.axisLookup[this.planeIds[dim][0]];
        const basisVectorY = this.axisLookup[this.planeIds[dim][1]];
        const basis = ort.stack([basisVectorX, basisVectorY]);

        for (let i = 0; i < 4; i++) {
            const oldVector = this.axisLookup[this.planeIds[dim][i]];
            this.axisLookup[this.planeIds[dim][i]] = await aguileraPerezRotation(oldVector, ort.transpose(basis), theta);
        }
    }

    async turnRightAngle(dim: number, theta = Math.PI / 2) {
        await this.turnLeftAngle(dim, -theta);
    }
}

class GameObject {
    nDims: number;
    pos: ort.Tensor;
    vel: NDVector;

    constructor(nDims: number) {
        this.nDims = nDims;
        this.pos = ort.zeros([nDims]);
        this.vel = new NDVector(nDims);
    }

    async facing(pos: ort.Tensor): Promise<ort.Tensor> {
        const norms = ort.stack(Object.values(this.vel.axisLookup));
        const diff = ort.sub(pos, this.pos);
        const repeatedDiff = diff.repeat(this.nDims * 2, 1);
        return ort.cosineSimilarity(repeatedDiff, norms);
    }

    async batchFacing(pos: ort.Tensor): Promise<ort.Tensor> {
        const results = await Promise.all(
            Array.from({ length: pos.dims[0] }, (_, i) => this.facing(pos.get(i)))
        );
        return ort.stack(results);
    }

    async distanceFromBounds(scale: number): Promise<ort.Tensor> {
        const norms = ort.stack(Object.values(this.vel.axisLookup));
        return ort.sum(ort.sub(ort.abs(ort.mul(norms, scale)), ort.abs(ort.mul(this.pos, norms))));
    }

    async batchDistanceFromBounds(pos: ort.Tensor): Promise<ort.Tensor> {
        return ort.cdist(this.pos.unsqueeze(0), pos);
    }
}

async function main() {
    const food = ort.tensor([1, 9, 0], 'float32');
    const obj = new GameObject(3);
    const vec = new NDVector(3);

    console.log(Object.values(vec.axisLookup));
    await vec.turnLeftAngle(0);
    console.log(Object.values(vec.axisLookup));
    await vec.turnLeftAngle(0);
    console.log(Object.values(vec.axisLookup));
}

main().catch(console.error);
