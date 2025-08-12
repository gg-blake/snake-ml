import { readLogs, writeLogs } from "@/app/actions";
import * as papaparse from "papaparse";

var timingsHistorical: TimingDetails = {};

export async function updateLogFile(id?: number) {
    const timings = getTimes();

    const data = await readLogs(id);
    if (typeof data !== "string") {
        throw data;
    }

    const appendCol = (csv: string[][], header: string) =>
        csv.map((row: string[], index: number) => {
            return [...row, index == 0 ? header : "0.0"];
        });

    papaparse.parse<string[]>(data, {
        complete: async (parsed) => {
            let csv = parsed.data;
            let headers: string[] = csv[0];
            csv = [...csv, Array.from({ length: headers.length }, () => "0.0")];
            const rowIndex = csv.length - 1;

            for (const key of Object.keys(timings)) {
                if (!headers.includes(key)) {
                    csv = appendCol(csv, key);
                    headers.push(key);
                }
                
                const colIndex = headers.indexOf(key);
                const val = timings[key].toString();
                csv[rowIndex][colIndex] = val;
            }

            const out = papaparse.unparse(csv, {
                skipEmptyLines: true
            });

            
            await writeLogs(out, id);
        },
    });
}

interface TimingDetails {
    [key: string]: number[];
}

export function getTimes() {
    const avgTimes: { [key: string]: number } = {};
    for (const label of Object.keys(timingsHistorical)) {
        let total = 0;
        avgTimes[label] = 0;
        if (timingsHistorical[label].length == 0) continue;
        for (const time of timingsHistorical[label]) {
            total = total + time;
        }
        avgTimes[label] = total / timingsHistorical[label].length;
    }
    return avgTimes;
}

export default class Logging {
    constructor(debugMode?: boolean) {
        if (!debugMode) return;
        const thisProto = Object.getPrototypeOf(this);
        const baseProto = Logging.prototype;

        const methodNames = Object.getOwnPropertyNames(thisProto).filter(
            (name) =>
                name !== "constructor" &&
                typeof (this as any)[name] === "function" &&
                baseProto[name as keyof Logging] === undefined && // not from base (Logging)
                !name.startsWith("_"), // exclude methods starting with "_"
        );

        for (const name of methodNames) {
            const originalMethod = (this as any)[name] as (
                ...args: any[]
            ) => any;

            const label = `${this.constructor.name}.${String(name)}`;
            if (!Object.keys(timingsHistorical).includes(label)) {
                timingsHistorical[label] = [];
            }
            (this as any)[name] = (...args: any[]) => {
                const start = performance.now();
                let end: number;

                const result = originalMethod.apply(this, args);

                if (result instanceof Promise) {
                    return result.finally(() => {
                        const end = performance.now();
                        timingsHistorical[label].push(end - start);
                    });
                } else {
                    const end = performance.now();
                    timingsHistorical[label].push(end - start);
                    return result;
                }
            };
        }
    }
    
    static reset() {
        for (const key of Object.keys(timingsHistorical)) {
            timingsHistorical[key] = [];
        }
    }

    log(...message: any[]) {
        (this as any).emit("info", ...message);
    }

    warn(...message: any[]) {
        (this as any).emit("warn", ...message);
    }

    error(...message: any[]) {
        (this as any).emit("error", ...message);
    }

    emit(messageType: "debug" | "warn" | "info" | "error", ...message: any[]) {
        console[messageType](
            `[${(this as any).constructor.name}]:`,
            ...message,
        );
    }
}
