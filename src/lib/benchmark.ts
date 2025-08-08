import Logging, { updateLogFile } from "./logger";
import main from "./main";
import { Model, ModelState } from "./model/model";
import { modelConfig, Trainer, TrainerState, trainingConfig } from "./model/trainer";
import { Mountable } from "./model/utils/types";
import { Renderer } from "./renderer/renderer";
import { initContext } from "./util";

interface BenchmarkConfig {
    numberOfSamples: number;
    startBatchSize: number;
    endBatchSize: number;
    batchInterval: number;
}

const benchmarkConfig = {
    numberOfSamples: 20,
    startBatchSize: 0,
    endBatchSize: 2000,
    batchInterval: 100,
};

function reset(mountedObject: Mountable | null) {
    if (!mountedObject) return;
    mountedObject.unmount();
}

function setupTrainer(trainer: Trainer | null, modelShape: [number, number, number]) {
    if (trainer !== null) {
        throw new Error("Trainer already mounted. Please call reset() before calling setupTrainer().")
    }
    const trainerState = new TrainerState();
    const model = new Model({
        ...modelConfig,
        batchInputShape: modelShape
    }, true)
    const modelState = new ModelState(model);
    trainer = new Trainer(modelState, trainerState, trainingConfig, true);
    return trainer
}

function setupRenderer(renderer: Renderer | null, canvas: HTMLCanvasElement) {
    if (renderer !== null) {
        throw new Error("Renderer already mounted. Please call reset() before calling setupRenderer().")
    }
    renderer = new Renderer(canvas, {
        verbose: true,
        debug: true,
    });
    
    return renderer
}

const logger = new Logging();

export default function benchmark(
    canvas: HTMLCanvasElement,
    config: BenchmarkConfig | undefined = benchmarkConfig
) {
    const [_, T, C] = modelConfig.batchInputShape! as number[];
    var trainer: Trainer | null;
    var renderer: Renderer | null;
    const sample = (b: number) => {
        Logging.reset();
        reset(trainer);
        trainer = null;
        reset(renderer);
        renderer = null;
        
        
        
        if (b >= config.endBatchSize) return;
        trainer = setupTrainer(trainer, [b > 0 ? b : 1, T, C]);
        renderer = setupRenderer(renderer, canvas);
        let sampleCount = 0;
        const render = (now: number) => {
            if (sampleCount == config.numberOfSamples) {
                logger.log(`benchmark collected [batches: ${b > 0 ? b : 1}, samples: ${config.numberOfSamples}]`);
                updateLogFile()
                    .then(() => sample(b + config.batchInterval))
                
                return;
            }

            main(renderer!, trainer!, now);
            sampleCount++;
            requestAnimationFrame(render);
        }

        initContext(canvas)
            .then(() => requestAnimationFrame(render))
    }
    
    sample(config.startBatchSize);
    logger.log(`Benchmarking complete!\nStats were written to ./public/logger-times.csv`);
}
