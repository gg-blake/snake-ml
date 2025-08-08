import * as tf from "@tensorflow/tfjs";
import { Trainer } from "./model/trainer";
import { Renderer } from "./renderer/renderer";
import Logging, { getTimes, updateLogFile } from "./logger";
import { loadTensorData } from "./util";

function test(renderer: Renderer, trainer: Trainer, now: number) {
    const glData = {
        gl: renderer._gl,
        program: renderer.program,
    };

    const instanceCount = 1000;

    const posData = loadTensorData(
        tf.randomUniform([instanceCount, 4]).mul(5),
        glData,
    );

    const colorData = loadTensorData(
        tf
            .randomUniform([instanceCount, 3])
            .concat(tf.ones([instanceCount, 1]), 1),
        glData,
    );

    renderer.render(posData.texture!, colorData.texture!, instanceCount);

    console.log("Number of undisposed tensors:", tf.memory().numTensors);
    // cleanup tensors
    posData.tensorRef.dispose();
    colorData.tensorRef.dispose();
}

const createSnakeLengthMask = (
    indices: tf.Tensor1D,
    maxLength: number,
): tf.Tensor2D =>
    tf.tidy(() => {
        const [B] = indices.shape;
        const stepSize = 1;
        const dtype = "float32";
        const arange = tf.range(0, maxLength, stepSize, dtype).tile([B]);
        const indicesTiled = indices
            .reshape([B, 1])
            .tile([1, maxLength])
            .reshape([B * maxLength]);
        const diff = arange.sub(indicesTiled);
        const mask = diff.less(0).reshape<tf.Tensor2D>([B, maxLength]);
        return mask;
    });

var rgb: tf.Tensor2D;
const maskSnakeRGBA = (indices: tf.Tensor1D, maxLength: number): tf.Tensor2D =>
    tf.tidy(() => {
        const [B] = indices.shape;
        const alpha = createSnakeLengthMask(indices, maxLength)
            .reshape([B * maxLength, 1])
            .cast("float32"); // 0.0 (false) / 1.0 (true)
        if (rgb == undefined) {
            rgb = tf.keep(tf.randomUniform([B, 3]));
        }
        const rgbTiled = rgb
            .reshape([B, 1, 3])
            .tile([1, maxLength, 1])
            .reshape([B * maxLength, 3]);
        const rgba = rgbTiled.concat(tf.ones([B * maxLength, 1]), 1);
        return rgba as tf.Tensor2D;
    });

const maskSnakeXYZ = (
    indices: tf.Tensor1D,
    maxLength: number,
    position: tf.Tensor2D,
) =>
    tf.tidy(() => {
        const [B] = indices.shape;
        const offset = createSnakeLengthMask(indices, maxLength)
            .reshape([B * maxLength, 1])
            .logicalNot()
            .cast("float32")
            .mul(1e5);
        const offsetTiled = offset.tile([1, 3]);
        const emptyAlphaChannel = tf.ones([B * maxLength, 1]);
        const offsetPosition = offsetTiled
            .add(position)
            .concat(emptyAlphaChannel, 1);
        return offsetPosition as tf.Tensor2D;
    });

var count = 0;
const logger = new Logging();

function main(renderer: Renderer, trainer: Trainer, now: number) {
    if (count == 100) {
        updateLogFile().then(() => {
            logger.log(
                "Benchmarking stats written to ./public/logging-times.csv",
            );
        });
    }
    now *= 0.001;

    trainer.step(); // Training step
    const snakeStartingLength = trainer.modelState.model._config.startingLength;
    const [B, T, C] = trainer.modelState.model.shape;
    const instanceCount = B * T;
    const scores =
        trainer.modelState.targetIndices.add<tf.Tensor1D>(snakeStartingLength);
    const historyReshaped = trainer.modelState.history.reshape<tf.Tensor2D>([
        instanceCount,
        C,
    ]);
    

    const cubePositions = maskSnakeXYZ(scores, T, historyReshaped);
    const cubeColors = maskSnakeRGBA(scores, T);
    const samplePositions = cubePositions.slice([0, 0], [T * 2, C]);
    const sampleColors = tf.randomUniform([T * 2, C]).concat(tf.ones([T * 2, 1]), 1);
    const glData = {
        gl: renderer._gl,
        program: renderer.program,
    };

    // Load tensor's textures and render the cubes
    const positionData = loadTensorData(cubePositions, glData);
    const colorData = loadTensorData(cubeColors, glData);
    renderer.render(positionData.texture!, colorData.texture!, instanceCount);

    // cleanup tensors
    positionData.tensorRef.dispose();
    colorData.tensorRef.dispose();
    count++;
}

export default main;
