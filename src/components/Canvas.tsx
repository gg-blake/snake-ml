"use client";
import { useRef, useEffect } from "react";
import { Renderer } from "../lib/renderer/renderer";
import { tidy, getBackend } from "@tensorflow/tfjs";
import { initContext } from "../lib/util";
import getTrainer, { Trainer } from "../lib/model/trainer";

const verbose = true;

export function Canvas({
    action,
}: {
    action: (renderer: Renderer, trainer: Trainer, now: number) => void;
}) {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const rendererRef = useRef<Renderer | null>(null);
    const trainerRef = useRef<Trainer | null>(null);

    const resizeCanvas = () => {
        const canvas = canvasRef.current;
        if (canvas) {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        }
    };

    useEffect(() => {
        // Set initial size
        resizeCanvas();

        // Resize on window resize
        window.addEventListener("resize", resizeCanvas);

        let initRender = false;
        if (!rendererRef.current) {
            initRender = true;
            initCanvas().then(() => {
                trainerRef.current = getTrainer();
                requestAnimationFrame(render);
            });
        } else {
            requestAnimationFrame(render);
        }

        return () => {
            // Remove listener
            window.removeEventListener("resize", resizeCanvas);

            // Prevent reregistration of tfjs backend on second mount
            if (initRender || !rendererRef.current) return;
            rendererRef.current.unmount(verbose);
        };
    }, []);

    const initCanvas = async () => {
        if (!canvasRef.current) return; // Prevent reregistration of tfjs backend on second mount

        const canvas = canvasRef.current;
        const renderer = new Renderer(canvas, verbose);
        rendererRef.current = renderer;

        initContext(canvas);
    };

    const render = (now: number) => {
        if (!rendererRef.current || !trainerRef.current) return;
        if (getBackend() != "custom-webgl") return;
        const renderer = rendererRef.current;
        const trainer = trainerRef.current;

        tidy(() => action(renderer, trainer, now));

        requestAnimationFrame(render);
    };

    return (
        <canvas
            ref={canvasRef}
            style={{
                display: "block",
                position: "absolute",
                top: 0,
                left: 0,
                width: "100vw",
                height: "100vh",
            }}
        />
    );
}
