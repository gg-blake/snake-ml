import { FolderApi, Pane, TabPageApi, BindingParams } from "tweakpane";
import { useEffect, useRef } from "react";
import { NEATRenderer } from "./renderer/renderer";
import * as THREE from 'three';
import GameLayer, * as GL from "./lib/layers/gamelayer";
import { NEATConfig } from "./lib/optimizer";
import { model } from "@tensorflow/tfjs";
import { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';

import { Inter } from 'next/font/google';
const inter = Inter({ subsets: ['latin'], weight: "variable" })
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
    model: GL.Config & LayerArgs;
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
        ttl: 20,
        batchInputShape: [100, 10, 3],
        startingLength: 5,
        boundingBoxLength: 30,
        units: 24,
        fitnessGraphParams: {
            a: 10,
            b: 1.5,
            c: 4,
            min: -1,
            max: 1
        },
        dtype: 'float32'
    },
    trainer: {
        mutationFactor: 0.1,
        mutationRate: 0
    }
}

var pendingSettings = settings;

export interface Stats {
    maxFitness: number;
    maxFitnessGlobal: number;
    maxTargetIndex: number;
    maxTargetIndexGlobal: number;
    framesToRestart: number;
    timeToRestart: number;
    fps: number;
    epochCount: number;
}

export var stats: Stats = {
    maxFitness: settings.model.ttl,
    maxFitnessGlobal: settings.model.ttl,
    maxTargetIndex: 0,
    maxTargetIndexGlobal: 0,
    framesToRestart: settings.model.ttl,
    timeToRestart: 0,
    fps: 0,
    epochCount: 0
}


settings.trainer.mutationRate = 1 / settings.model.batchInputShape![0]!;
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
        addModelParameterBinding(tab.pages[0], 'ttl', { step: 10, min: 50, max: 300, label: 'Time to Live' }, endRequest);
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
            max: settings.model.ttl,
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

        pane.addBinding(stats, 'maxTargetIndex', {
            readonly: true,
            view: 'graph',
            min: 0,
            max: settings.model.batchInputShape![1]!,
            label: "Max Target Index Current Epoch"
        })

        pane.addBinding(stats, 'maxTargetIndex', {
            readonly: true,
            label: "Max Target Index Current Epoch"
        })

        pane.addBinding(stats, 'maxTargetIndexGlobal', {
            readonly: true,
            view: 'graph',
            min: 0,
            max: settings.model.batchInputShape![1]!,
            label: "Max Target Index All-time"
        })

        pane.addBinding(stats, 'maxTargetIndexGlobal', {
            readonly: true,
            label: "Max Target Index All-time"
        })

        return () => {
            console.log('Ref unmounted');
        };
    }

    return (
        <div className="absolute right-0 top-0 m-2 z-100">
            <div className={`${inter.className} z-100 border-primary-foreground border-[1px] rounded-lg`} ref={paneRef}>

            </div>
        </div>
    )
}