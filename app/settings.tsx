import { FolderApi, Pane, TabPageApi } from "tweakpane";
import { useEffect, useRef } from "react";
import Renderer from "./lib/renderer";
import * as THREE from 'three';
import GameLayer, { GameLayerConfig } from "./lib/layers/gamelayer";

export interface RendererConfig {
    "Primary Current Snake Color": string;
    "Secondary Current Snake Color": string;
    "Primary Next Snake Color": string;
    "Secondary Next Snake Color": string;
    "Primary Food Color": string;
    "Secondary Food Color": string;
}

// TODO: Move this interface to a dedicated file
interface DifferentialEvolutionTrainerConfig {
    CR: number; // Crossover probability
    F: number; // Differential weight
}



export interface Settings {
    renderer: RendererConfig;
    model: GameLayerConfig;
    trainer: DifferentialEvolutionTrainerConfig;
}

export var settings: Settings = {
    renderer: {
        "Primary Current Snake Color": `#${Renderer.primaryGameObjectMaterialCurrent.color.getHexString()}`,
        "Secondary Current Snake Color": `#${Renderer.secondaryGameObjectMaterialCurrent.color.getHexString()}`,
        "Primary Next Snake Color": `#${Renderer.primaryGameObjectMaterialNext.color.getHexString()}`,
        "Secondary Next Snake Color": `#${Renderer.secondaryGameObjectMaterialNext.color.getHexString()}`,
        "Primary Food Color": `#${Renderer.primaryFoodMaterial.color.getHexString()}`,
        "Secondary Food Color": `#${Renderer.secondaryFoodMaterial.color.getHexString()}`,
    },
    model: {
        TTL: 300,
        B: 5,
        T: 10,
        C: 3,
        startingLength: 5,
        boundingBoxLength: 30,
        units: 40,
        fitnessGraphParams: {
            a: 10,
            b: 1.5,
            c: 4,
            min: -1,
            max: 1
        }
    },
    trainer: {
        CR: 0.9,
        F: 0.8,
    }
}



function addColorBinding(pane: Pane | TabPageApi | FolderApi, key: keyof Settings['renderer'], material: THREE.MeshStandardMaterial) {
    const b = pane.addBinding(settings.renderer, key, {
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
        tab.pages[0].addBinding(settings.model, 'TTL', { step: 10, min: 50, max: 300, label: 'Time to Live' });
        tab.pages[0].addBinding(settings.model, 'B', { step: 1, min: 4, max: 1000, label: 'Batch Size' });
        tab.pages[0].addBinding(settings.model, 'T', { step: 1, min: 10, max: 100, label: 'Number of Food' });
        tab.pages[0].addBinding(settings.model, 'C', { step: 1, min: 2, max: 15, label: 'Number of Dimensions' });
        tab.pages[0].addBinding(settings.trainer, 'CR', { step: 0.01, min: 0, max: 1, label: 'Crossover Probability' });
        tab.pages[0].addBinding(settings.trainer, 'F', { step: 0.01, min: 0, max: 1, label: 'Differential Weight'});
        tab.pages[0].addBinding(settings.model, 'startingLength', { step: 1, min: 1, max: 100, label: 'Starting Snake Length' });
        tab.pages[0].addBinding(settings.model, 'boundingBoxLength', { step: 5, min: 5, max: 1000, label: 'Bound Box Length' });
        tab.pages[0].addBinding(settings.model, 'units', { step: 1, min: 1, max: 300, label: 'Number of Hidden Layer Nodes' });

        const fitnessGraphParamsFolder = tab.pages[0].addFolder({
            'title': 'Fitness Graph'
        })

        fitnessGraphParamsFolder.addBinding(settings.model.fitnessGraphParams, 'a', { step: 0.0001, min: -50, max: 50 });
        fitnessGraphParamsFolder.addBinding(settings.model.fitnessGraphParams, 'b', { step: 0.0001, min: -50, max: 50 });
        fitnessGraphParamsFolder.addBinding(settings.model.fitnessGraphParams, 'c', { step: 0.0001, min: -50, max: 50 });
        fitnessGraphParamsFolder.addBinding(settings.model.fitnessGraphParams, 'min');
        fitnessGraphParamsFolder.addBinding(settings.model.fitnessGraphParams, 'max');

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