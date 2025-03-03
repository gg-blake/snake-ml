import { Pane } from "tweakpane";
import { useEffect, useRef } from "react";

export interface Settings {
    "Time To Live": number;
    "Batch Size": number; // Population size
    "Number of Food": number; // Number of food loaded
    "Number of Dimensions": number; // Dimensionality game space
    "Crossover Probabilty": number; // Crossover probability
    "Differential Weight": number; // Differential weight
    "Starting Snake Length": number; // Starting score
    "Bound Box Length": number; // Bound box side length
    "Number of Hidden Layer Nodes": number; // Dimensionality of hidden layer(s)
}

export var settings: Settings = {
    "Time To Live": 100,
    "Batch Size": 50,
    "Number of Food": 10,
    "Number of Dimensions": 3,
    "Crossover Probabilty": 0.9,
    "Differential Weight": 0.8,
    "Starting Snake Length": 5,
    "Bound Box Length": 30,
    "Number of Hidden Layer Nodes": 40
}

export function SettingsPane() {
    const paneRef = useRef(null);
    const isMounted = useRef<boolean>(false);

    useEffect(() => {
        if (isMounted.current) return;
        isMounted.current = true;

        onMount();
    }, [])

    const onMount = () => {
        if (!paneRef.current) return;
        if (!isMounted.current) return;

        const pane = new Pane({ container: paneRef.current });
        pane.addBinding(settings, 'Time To Live', {
            step: 10,
            min: 50,
            max: 300
        });
        pane.addBinding(settings, 'Batch Size', {
            step: 1,
            min: 4,
            max: 1000
        });
        pane.addBinding(settings, 'Number of Food', {
            step: 1,
            min: 10,
            max: 100
        });
        pane.addBinding(settings, 'Number of Dimensions', {
            step: 1,
            min: 2,
            max: 15
        });
        pane.addBinding(settings, 'Crossover Probabilty', {
            step: 0.01,
            min: 0,
            max: 1
        });
        pane.addBinding(settings, 'Differential Weight', {
            step: 0.01,
            min: 0,
            max: 1
        });
        pane.addBinding(settings, 'Starting Snake Length', {
            step: 1,
            min: 1,
            max: 100
        });
        pane.addBinding(settings, 'Bound Box Length', {
            step: 5,
            min: 5,
            max: 1000
        });
        pane.addBinding(settings, 'Number of Hidden Layer Nodes', {
            step: 1,
            min: 1,
            max: 300
        });

        return () => {
            console.log('Ref unmounted');
        };
    }

    return (
        <div className="absolute right-0 m-2">
            <div ref={paneRef}>

            </div>
        </div>
    )
}