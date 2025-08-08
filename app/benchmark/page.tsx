"use client";
import { GameMenu } from "@/src/components/GameMenu";
import { Canvas } from "@/src/components/Canvas";
import { modelConfig, trainingConfig } from "@/src/lib/model/trainer";
import GameStartContext from "@/src/components/GameStartContextProvider";
import { useContext, useRef } from "react";
import main from "@/src/lib/main";
import { Badge } from "@/src/components/ui/badge";

export default function Home() {
    const { gameStarted } = useContext(GameStartContext);
    const fpsElementRef = useRef<HTMLDivElement>(null!);

    return (
        <div className="w-screen h-screen flex flex-row-reverse">
            { gameStarted && <Canvas action={main} fpsElementRef={fpsElementRef} mode="benchmark" /> }
            <GameMenu config={modelConfig} />
            <Badge className="absolute top-3 right-3 flex" variant="secondary"><div ref={fpsElementRef}></div></Badge>
        </div>
    );
}