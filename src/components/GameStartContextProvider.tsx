"use client"
import { createContext, Dispatch, SetStateAction, useState } from "react";

interface GameStartContextI {
    gameStarted: boolean;
    setGameStarted: Dispatch<SetStateAction<boolean>>;
}

const GameStartContext = createContext<GameStartContextI>({
    gameStarted: false,
    setGameStarted: () => {}
});

export function GameStartProvider({ children }: { children: React.ReactNode }) {
    const [gameStarted, setGameStarted] = useState(false);

    return (
        <GameStartContext.Provider value={{ gameStarted, setGameStarted }}>
            {children}
        </GameStartContext.Provider>
    );
}

export default GameStartContext;
