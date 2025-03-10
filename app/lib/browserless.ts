import * as tf from '@tensorflow/tfjs';
import { settings, Settings } from '../settings';
import { ModelState, DifferentialEvolutionTrainer, arraySync, dispose, projectDirectionToBounds, trainer } from './model';
import '@tensorflow/tfjs-backend-webgl';

var killRequest = false;

export default function main() {
    const initializer = (): ModelState<DifferentialEvolutionTrainer> => {
        const B = settings["Batch Size"];
        const T = settings["Number of Food"];
        const C = settings["Number of Dimensions"];
        const CR = settings["Crossover Probabilty"];
        const F = settings["Differential Weight"];
        const G = settings["Bound Box Length"];

        // Initialize Models
        const currentModel = new DifferentialEvolutionTrainer({ batchInputShape: [B, T, C] });
        const nextModel = new DifferentialEvolutionTrainer({ batchInputShape: [B, T, C] });
        currentModel.build();
        nextModel.build();
        nextModel.buildFrom(currentModel, CR, F);

        return [
            currentModel,
            nextModel
        ]
    }

    const preAnimate = (...args: ModelState<DifferentialEvolutionTrainer>): boolean => { 
        console.time('frame');
        if (killRequest) {
            killRequest = false;
            return false
        };

        const CR = settings["Crossover Probabilty"];
        const F = settings["Differential Weight"];

        if (args[0].state.active.sum().arraySync() === 0 && args[1].state.active.sum().arraySync() === 0) {
            tf.tidy(() => {
                
                args[0].buildFromCompare(args[1]);
                
                args[1].buildFrom(args[0], CR, F);
                
                args[0].resetState();
                args[1].resetState();
            })
        };

        return true;
    }

    const postAnimate = (animate: () => void, ...args: ModelState<DifferentialEvolutionTrainer>) => {
        console.timeEnd('frame');
        animate();
    } 

    trainer(
        initializer(),
        preAnimate,
        postAnimate
    );
};

main();