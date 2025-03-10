import { FolderApi, Pane, TabPageApi } from "tweakpane";
import { useEffect, useRef } from "react";
import Renderer from "./lib/renderer";
import * as THREE from 'three';
import { fitnessGraphParams } from "./lib/model";

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
    "Primary Current Snake Color": string;
    "Secondary Current Snake Color": string;
    "Primary Next Snake Color": string;
    "Secondary Next Snake Color": string;
    "Primary Food Color": string;
    "Secondary Food Color": string;
    "Fitness Graph Params": fitnessGraphParams;
}

export var settings: Settings = {
    "Time To Live": 300,
    "Batch Size": 200,
    "Number of Food": 10,
    "Number of Dimensions": 3,
    "Crossover Probabilty": 0.9,
    "Differential Weight": 0.8,
    "Starting Snake Length": 5,
    "Bound Box Length": 30,
    "Number of Hidden Layer Nodes": 40,
    "Primary Current Snake Color": `#${Renderer.primaryGameObjectMaterialCurrent.color.getHexString()}`,
    "Secondary Current Snake Color": `#${Renderer.secondaryGameObjectMaterialCurrent.color.getHexString()}`,
    "Primary Next Snake Color": `#${Renderer.primaryGameObjectMaterialNext.color.getHexString()}`,
    "Secondary Next Snake Color": `#${Renderer.secondaryGameObjectMaterialNext.color.getHexString()}`,
    "Primary Food Color": `#${Renderer.primaryFoodMaterial.color.getHexString()}`,
    "Secondary Food Color": `#${Renderer.secondaryFoodMaterial.color.getHexString()}`,
    "Fitness Graph Params": {
        a: 10,
        b: 1.5,
        c: 4,
        min: -1,
        max: 1
    }
}

function addColorBinding(pane: Pane | TabPageApi | FolderApi, key: keyof Settings, material: THREE.MeshStandardMaterial) {
    const b = pane.addBinding(settings, key, {
        picker: 'inline',
        expanded: true,
    })
    b.on('change', (ev) => {
        material.color.lerp(new THREE.Color(ev.value as number | string), 0.5)
    })
}

function addTrainingParameterBinding(pane: Pane | TabPageApi | FolderApi, key: keyof Settings, step: number, min: number, max: number) {
    pane.addBinding(settings, key, {
        step: step,
        min: min,
        max: max
    });
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
        const tab = pane.addTab({
            pages: [
                { title: 'Parameters' },
                { title: 'Visual' },
            ],
        });
        tab.pages[0].addBinding(settings, 'Time To Live', { step: 10, min: 50, max: 300 });
        tab.pages[0].addBinding(settings, 'Batch Size', { step: 1, min: 4, max: 1000 });
        tab.pages[0].addBinding(settings, 'Number of Food', { step: 1, min: 10, max: 100 });
        tab.pages[0].addBinding(settings, 'Number of Dimensions', { step: 1, min: 2, max: 15 });
        tab.pages[0].addBinding(settings, 'Crossover Probabilty', { step: 0.01, min: 0, max: 1 });
        tab.pages[0].addBinding(settings, 'Differential Weight', { step: 0.01, min: 0, max: 1 });
        tab.pages[0].addBinding(settings, 'Starting Snake Length', { step: 1, min: 1, max: 100 });
        tab.pages[0].addBinding(settings, 'Bound Box Length', { step: 5, min: 5, max: 1000 });
        tab.pages[0].addBinding(settings, 'Number of Hidden Layer Nodes', { step: 1, min: 1, max: 300 });

        const fitnessGraphParamsFolder = tab.pages[0].addFolder({
            'title': 'Fitness Graph'
        })

        fitnessGraphParamsFolder.addBinding(settings["Fitness Graph Params"], 'a', { step: 0.0001, min: -50, max: 50 });
        fitnessGraphParamsFolder.addBinding(settings["Fitness Graph Params"], 'b', { step: 0.0001, min: -50, max: 50 });
        fitnessGraphParamsFolder.addBinding(settings["Fitness Graph Params"], 'c', { step: 0.0001, min: -50, max: 50 });
        fitnessGraphParamsFolder.addBinding(settings["Fitness Graph Params"], 'min');
        fitnessGraphParamsFolder.addBinding(settings["Fitness Graph Params"], 'max');

        const visualColorFolder = tab.pages[1].addFolder({
            'title': 'Colors'
        })
        

        addColorBinding(visualColorFolder, "Primary Current Snake Color", Renderer.primaryGameObjectMaterialCurrent);
        addColorBinding(visualColorFolder, "Secondary Current Snake Color", Renderer.secondaryGameObjectMaterialCurrent);
        addColorBinding(visualColorFolder, "Primary Next Snake Color", Renderer.primaryGameObjectMaterialNext);
        addColorBinding(visualColorFolder, "Secondary Next Snake Color", Renderer.secondaryGameObjectMaterialNext);
        addColorBinding(visualColorFolder, "Primary Food Color", Renderer.primaryFoodMaterial);
        addColorBinding(visualColorFolder, "Secondary Food Color", Renderer.secondaryFoodMaterial);

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