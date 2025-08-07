"use client";
import {
    ChangeEvent,
    ChangeEventHandler,
    MutableRefObject,
    RefObject,
    useContext,
    useEffect,
    useRef,
    useState,
} from "react";
import { Input } from "@/src/components/ui/input";
import { Slider } from "@/src/components/ui/slider";
import { LayerArgs } from "@tensorflow/tfjs-layers/dist/engine/topology";
import {
    Card,
    CardContent,
    CardDescription,
    CardFooter,
    CardHeader,
    CardTitle,
} from "@/src/components/ui/card";
import { Button } from "./ui/button";
import GameStartContext from "./GameStartContextProvider";
import { Config } from "../lib/model/utils/types";

type MenuPropertyObject = { [key: string | symbol]: number };

function MenuProperty<T = Settings["model"]>({
    children,
    onChange,
    defaultValue,
    step,
    min,
    max,
}: {
    children: string;
    onChange: (value: number) => void;
    defaultValue: number;
    step: number;
    min: number;
    max: number;
}) {
    const [value, setValue] = useState<[number]>([defaultValue]);

    const onSliderChange = (value: number[]) => {
        onChange(value[0]);
        setValue([value[0]]);
    };

    const onInputChange = (e: ChangeEvent<HTMLInputElement>) => {
        let newValue = Math.round(parseFloat(e.target.value) * step) / step;
        if (newValue < min) {
            newValue = min;
        } else if (newValue > max) {
            newValue = max;
        }

        onChange(newValue);
        setValue([newValue]);
    };

    return (
        <div className="flex flex-col w-full h-auto gap-2 p-1">
            <div className="text-xs">{children}</div>
            <div className="flex flex-row w-full h-auto gap-3 text-xs">
                <Input
                    onChange={onInputChange}
                    value={value[0]}
                    type="number"
                    className="w-[70px] text-right text-xs"
                />
                <Slider
                    defaultValue={value}
                    min={min}
                    max={max}
                    step={step}
                    className="w-full"
                    value={value}
                    onValueChange={onSliderChange}
                />
            </div>
        </div>
    );
}

/*
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
        max: 1,
    },
    dtype: "float32",
}
*/

export function GameMenu({ config }: { config: Config & LayerArgs }) {
    const { gameStarted, setGameStarted } = useContext(GameStartContext);


    return (
        <Card className="absolute top-3 left-3 flex flex-col w-[30vw] h-auto origin-top-left scale-75">
            <CardHeader>
                <CardTitle>Model Settings</CardTitle>
                <CardDescription>
                    Adjust the model parameters for training
                </CardDescription>
            </CardHeader>
            <CardContent>
                {/* Model Properties */}
                <MenuProperty
                    defaultValue={config.batchInputShape![0]! | 0}
                    onChange={(value: number) =>
                        (config.batchInputShape![0]! = value)
                    }
                    min={10}
                    max={1000}
                    step={1}
                >
                    Batch Size
                </MenuProperty>
                <MenuProperty
                    defaultValue={config.batchInputShape![1]! | 0}
                    onChange={(value: number) =>
                        (config.batchInputShape![1]! = value)
                    }
                    min={10}
                    max={100}
                    step={1}
                >
                    Max Snake Length
                </MenuProperty>
                <MenuProperty
                    defaultValue={config.batchInputShape![2]! | 0}
                    onChange={(value: number) =>
                        (config.batchInputShape![2]! = value)
                    }
                    min={1}
                    max={4}
                    step={1}
                >
                    Number of Dimensions
                </MenuProperty>
                <MenuProperty
                    defaultValue={config.ttl | 0}
                    onChange={(value: number) => (config.ttl = value)}
                    min={10}
                    max={2000}
                    step={1}
                >
                    Snake Time to Live (steps)
                </MenuProperty>
                <MenuProperty
                    defaultValue={config.startingLength | 0}
                    onChange={(value: number) =>
                        (config.startingLength = value)
                    }
                    min={3}
                    max={200}
                    step={1}
                >
                    Snake Starting Length
                </MenuProperty>
                <MenuProperty
                    defaultValue={config.boundingBoxLength | 0}
                    onChange={(value: number) =>
                        (config.boundingBoxLength = value)
                    }
                    min={5}
                    max={2000}
                    step={10}
                >
                    Bounding Box Length
                </MenuProperty>
                <MenuProperty
                    defaultValue={config.units | 0}
                    onChange={(value: number) => (config.units = value)}
                    min={2}
                    max={500}
                    step={1}
                >
                    Layer Units
                </MenuProperty>
                <Button
                    variant={gameStarted ? "destructive" : "default"}
                    onClick={() => setGameStarted(!gameStarted)}
                >
                    {gameStarted ? "Shutdown" : "Start"}
                </Button>
            </CardContent>
        </Card>
    );
}
