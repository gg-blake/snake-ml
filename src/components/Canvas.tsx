"use client";
import { useRef, useEffect, RefObject } from "react";
import { Renderer } from "../lib/renderer/renderer";
import { tidy, getBackend } from "@tensorflow/tfjs";
import { initContext } from "../lib/util";
import getTrainer, { Trainer } from "../lib/model/trainer";
import { useContext } from "react";
import GameStartContext from "./GameStartContextProvider";
import benchmark from "../lib/debug_tools/benchmark";

const verbose = true;
const debug = true;

let then = 0;
export function Canvas({
    action,
    fpsElementRef,
    mode = "benchmark",
}: {
    action: (renderer: Renderer, trainer: Trainer, now: number) => void;
    fpsElementRef: RefObject<HTMLDivElement>;
    mode?: "default" | "benchmark";
}) {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const rendererRef = useRef<Renderer | null>(null);
    const trainerRef = useRef<Trainer | null>(null);
    const { gameStarted } = useContext(GameStartContext);

    const resizeCanvas = () => {
        const canvas = canvasRef.current;
        if (canvas) {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        }
    };

    useEffect(() => {
        if (!["default", "benchmark"].includes(mode)) {
            throw new Error("[Canvas]: invalid mode specified.");
        }
        if (!gameStarted) return;
        if (!canvasRef.current) return;
        // Set initial size
        resizeCanvas();

        // Resize on window resize
        window.addEventListener("resize", resizeCanvas);

        

        let initRender = false;
        if (!rendererRef.current) {
            initRender = true;
            initCanvas(canvasRef.current).then(() => {
                trainerRef.current = getTrainer(debug);
            });
        } else {
            if (mode == "benchmark") {
                benchmark(
                    canvasRef.current,
                );
                return;
            }
            requestAnimationFrame(render);
            console.log("no new renderer created");
        }

        return () => {
            // Remove listener
            window.removeEventListener("resize", resizeCanvas);

            // Prevent reregistration of tfjs backend on second mount
            if (initRender || !rendererRef.current || !trainerRef.current)
                return;
            rendererRef.current.unmount(verbose);
            trainerRef.current.unmount(verbose);
            rendererRef.current = null;
            trainerRef.current = null;
            console.log("renderer and trainer removed");
        };
    }, [gameStarted]);

    const initCanvas = async (canvas: HTMLCanvasElement) => {
        const renderer = new Renderer(canvas, {
            verbose: verbose,
            debug: debug,
        });
        rendererRef.current = renderer;

        initContext(canvas);
    };

    const render = (now: number) => {
        if (
            !rendererRef.current ||
            !trainerRef.current ||
            !fpsElementRef.current
        )
            return;
        if (getBackend() != "custom-webgl") return;
        const renderer = rendererRef.current;
        const trainer = trainerRef.current;
        now *= 0.001; // convert to seconds
        const deltaTime = now - then; // compute time since last frame
        then = now; // remember time for next frame
        const fps = 1 / deltaTime; // compute frames per second
        if (now % 2 > 1.9) {
            fpsElementRef.current.textContent = `FPS: ${fps.toFixed(1)}`; // update fps display
        }

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
