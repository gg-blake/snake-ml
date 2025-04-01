import { FolderApi, Pane, TabPageApi, BindingParams } from "tweakpane";
import { useEffect, useRef } from "react";
import { NEATRenderer } from "./lib/renderer";
import * as THREE from 'three';
import GameLayer, { GameLayerConfig } from "./lib/layers/gamelayer";
import { NEATConfig } from "./lib/optimizer";
import { model } from "@tensorflow/tfjs";

export interface RendererConfig {
    showProjections: boolean;
    showFitnessDelta: boolean;
    showNormals: boolean;
    showTargetRays: boolean;
    showBest: boolean;
    showTargetProjections: boolean;
    renderEpochInterval: number;
}

// TODO: Move this interface to a dedicated file
interface DifferentialEvolutionTrainerConfig {
    CR: number; // Crossover probability
    F: number; // Differential weight
}

export interface Settings {
    renderer: RendererConfig;
    model: GameLayerConfig;
    trainer: NEATConfig;
}

var settings: Settings = {
    renderer: {
        showProjections: false,
        showFitnessDelta: false,
        showNormals: false,
        showTargetRays: false,
        showBest: false,
        showTargetProjections: false,
        renderEpochInterval: 1
    },
    model: {
        TTL: 200,
        B: 100,
        T: 10,
        C: 3,
        startingLength: 5,
        boundingBoxLength: 30,
        units: 100,
        fitnessGraphParams: {
            a: 10,
            b: 1.5,
            c: 4,
            min: -1,
            max: 1
        }
    },
    trainer: {
        mutationFactor: 0.01,
        mutationRate: 0
    }
}

var pendingSettings = settings;

export interface Stats {
    maxFitness: number;
    maxFitnessGlobal: number;
    framesToRestart: number;
    timeToRestart: number;
    fps: number;
    epochCount: number;
}

export var stats: Stats = {
    maxFitness: settings.model.TTL,
    maxFitnessGlobal: settings.model.TTL,
    framesToRestart: settings.model.TTL,
    timeToRestart: 0,
    fps: 0,
    epochCount: 0
}


settings.trainer.mutationRate = 0.5;
pendingSettings.trainer.mutationRate = settings.trainer.mutationRate
export {settings, pendingSettings};

function addTrainingParameterBinding(pane: Pane | TabPageApi | FolderApi, key: keyof Settings["trainer"], params: BindingParams, killRequest: React.MutableRefObject<boolean>) {
    pane.addBinding(pendingSettings.trainer, key, params)
    .on('change', () => {
        killRequest.current! = true;
    })
}

function addModelParameterBinding(pane: Pane | TabPageApi | FolderApi, key: keyof Settings["model"], params: BindingParams, endRequest: React.MutableRefObject<boolean>) {
    pane.addBinding(pendingSettings.model, key, params)
    .on('change', () => {
        endRequest.current! = true;
    })
}

export function SettingsPane({ killRequest, endRequest }: { killRequest: React.MutableRefObject<boolean>, endRequest: React.MutableRefObject<boolean> }) {
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
        addModelParameterBinding(tab.pages[0], 'TTL', { step: 10, min: 50, max: 300, label: 'Time to Live' }, endRequest);
        addModelParameterBinding(tab.pages[0], 'B', { step: 1, min: 4, max: 1000, label: 'Batch Size' }, endRequest)
        addModelParameterBinding(tab.pages[0], 'T', { step: 1, min: 10, max: 100, label: 'Number of Food' }, endRequest);
        addModelParameterBinding(tab.pages[0], 'C', { step: 1, min: 2, max: 15, label: 'Number of Dimensions' }, endRequest);
        addModelParameterBinding(tab.pages[0], 'startingLength', { step: 1, min: 1, max: 100, label: 'Starting Snake Length' }, endRequest);
        addModelParameterBinding(tab.pages[0], 'boundingBoxLength', { step: 5, min: 5, max: 1000, label: 'Bound Box Length' }, endRequest);
        addModelParameterBinding(tab.pages[0], 'units', { step: 1, min: 1, max: 300, label: 'Number of Hidden Layer Nodes' }, endRequest);
        addTrainingParameterBinding(tab.pages[0], 'mutationRate', { step: 0.01, min: 0, max: 1, label: 'Mutation Rate' }, killRequest);
        addTrainingParameterBinding(tab.pages[0], 'mutationFactor', { step: 0.01, min: 0, max: 1, label: 'Mutation Factor' }, killRequest);
        

        const fitnessGraphParamsFolder = tab.pages[0].addFolder({
            'title': 'Fitness Graph'
        })

        fitnessGraphParamsFolder.addBinding(pendingSettings.model.fitnessGraphParams, 'a', { step: 0.0001, min: -50, max: 50 });
        fitnessGraphParamsFolder.addBinding(pendingSettings.model.fitnessGraphParams, 'b', { step: 0.0001, min: -50, max: 50 });
        fitnessGraphParamsFolder.addBinding(pendingSettings.model.fitnessGraphParams, 'c', { step: 0.0001, min: -50, max: 50 });
        fitnessGraphParamsFolder.addBinding(pendingSettings.model.fitnessGraphParams, 'min');
        fitnessGraphParamsFolder.addBinding(pendingSettings.model.fitnessGraphParams, 'max');

        tab.pages[1].addBinding(pendingSettings.renderer, 'showProjections', { label: 'Show Snake Projections'});
        tab.pages[1].addBinding(pendingSettings.renderer, 'showFitnessDelta', { label: 'Show Fitness Delta' });
        tab.pages[1].addBinding(pendingSettings.renderer, 'showNormals', { label: 'Show Velocity Vectors' });
        tab.pages[1].addBinding(pendingSettings.renderer, 'showTargetRays', { label: 'Show Target Rays' });
        tab.pages[1].addBinding(pendingSettings.renderer, 'showBest', { label: 'Show Best Performers' });
        tab.pages[1].addBinding(pendingSettings.renderer, 'showTargetProjections', { label: 'Show Target Projections' });

        
        pane.addBinding(stats, 'maxFitness', {
            readonly: true,
            view: 'graph',
            min: 0,
            max: 3000,
            label: "Max Fitness"
        });

        pane.addBinding(stats, 'maxFitness', {
            readonly: true,
            label: "Max Fitness"
        });

        pane.addBinding(stats, 'maxFitnessGlobal', {
            readonly: true,
            label: "Max Fitness All-time"
        });

        pane.addBinding(stats, 'framesToRestart', {
            readonly: true,
            view: 'graph',
            min: 0,
            max: settings.model.TTL,
            label: "Number of Frames Until Next Generation"
        });

        pane.addBinding(stats, 'framesToRestart', {
            readonly: true,
            label: "Estimated Seconds Until Next Generation"
        });

        pane.addBinding(stats, 'fps', {
            readonly: true,
            lable: "FPS"
        });

        pane.addBinding(stats, 'epochCount', {
            readonly: true,
            lable: "Current Epoch"
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