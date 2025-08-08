"use server";
import fs from "fs/promises"; // Use fs.promises for async operations

const filename = "./public/logger-times.csv";

export async function writeLogs(data: string): Promise<void | Error> {
    try {
        await fs.writeFile(filename, data);
        return;
    } catch (error: any) {
        return new Error(error.message);
    }
}

export async function readLogs(): Promise<string | Error> {
    try {
        const file = await fs.readFile(filename, 'utf8');
        return file;
    } catch (error: any) {
        return new Error(error.message);
    }
}

