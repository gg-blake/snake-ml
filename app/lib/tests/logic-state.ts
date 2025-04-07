import * as tf from '@tensorflow/tfjs';
import { Settings, settings } from '@/app/settings';
import InputNorm from '../layers/inputnorm';
import getModel from '../model';
import NEAT from '../optimizer';
import Logic from '../layers/logic';

const B = 5;
const C = 3;
const T = 10;

settings.trainer.mutationRate = 1 / settings.model.batchInputShape![0]!;

function main() {

    const model = getModel(settings.model, 123);

    const trainer = new NEAT(settings);
    trainer.resetState();
    trainer.target.print(true);

    for (let i = 0; i < 100; i++) {
        
        trainer.step(model);
    }
    
    
    const logic = model.getLayer("logic") as Logic;

    //logic.state.fitnessDelta.print(true);
}

/*
Important note for later:

There is a chance that the mlp is attempting to rotate on a plane that does not change the forward direction vector.
The plane indices include subsets of axis that do not necessarily include the forward-facing direction vector.
We must account for this concept when implementing the controls. 
This is likely the reason we get unwanted results from our snakes; it is attempting the turn on a plane orthoganol to its forward facing vector which
is why it could crash into the wall or completely ignore the food.

Potential remedies:
1. Projection onto appropriate planes of rotation
   -   Take all planes that include the forward direction vector and project the target onto all of those planes. 
   -   Then calculate the angle (cosine similarity) between each of the projected target vectors and the forward vector.
   -   NOTE: These projections maybe expensive since we apply projections to subsets of vectors increases larger than exponentially with large values of n (n=number of dimensions)
   
   -   (Think about this procedure. This is not fully thought out & fully-schizo post -> ) Reuse the raycast() function from calculateBounds() and in place of bounds use all vectors (except the forward vectors but each other vector must be part of forward vector in the plane indices) as normal vectors for the planes?
   */

main();