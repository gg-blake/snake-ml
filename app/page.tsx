"use client";
import { GameMenu } from "@/src/components/GameMenu";
import { Canvas } from "@/src/components/Canvas";
import { modelConfig, trainingConfig } from "@/src/lib/model/trainer";
import GameStartContext from "@/src/components/GameStartContextProvider";
import { useContext } from "react";
import main from "@/src/lib/main";

export default function Home() {
    const { gameStarted } = useContext(GameStartContext);

    return (
        <div className="w-screen h-screen flex flex-row-reverse">
            { gameStarted && <Canvas action={main} /> }
            <GameMenu config={modelConfig} />
        </div>
    );
}
