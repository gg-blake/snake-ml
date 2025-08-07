import * as tf from "@tensorflow/tfjs";
import { loadTensorTexture } from "./util";
import { Renderer } from "./renderer/renderer";
import { Trainer } from "./model/trainer";

function main(renderer: Renderer, trainer: Trainer, now: number) {
    now *= 0.001;

    trainer.step();

    const history = trainer.modelState.history;
    const [batchSize, timeSize, channelSize] = trainer.modelState.model.shape;
    const instanceCount = batchSize * timeSize;
    const historyReshaped = history.reshape([instanceCount, channelSize]);

    const alphaConcat = historyReshaped.concat(tf.ones([instanceCount, 1]), 1);
    const color = tf
        .tensor([0.5, 0.9, 0.1, 1.0], [1, 4], "float32")
        .tile([instanceCount, 1]);

    const glData = {
        gl: renderer._gl,
        program: renderer.program,
    };

    const posData = loadTensorTexture(alphaConcat, glData);
    const posTexture = posData.texture!;

    const colorData = loadTensorTexture(color, glData);
    const colorTexture = colorData.texture!;

    renderer.render(posTexture, colorTexture, instanceCount);

    console.log("Number of undisposed tensors:", tf.memory().numTensors);
    // cleanup tensors
    posData.tensorRef.dispose();
    colorData.tensorRef.dispose();
};

export default main;
