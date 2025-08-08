"use server";
import fs from "fs/promises"; // Use fs.promises for async operations

const filename = (id?: number) => `./public/logger-times${id ? id.toString() : ""}.csv`;

export async function writeLogs(data: string, id?: number): Promise<void | Error> {
    try {
        await fs.writeFile(filename(id), data);
        return;
    } catch (error: any) {
        return new Error(error.message);
    }
}

export async function readLogs(id?: number): Promise<string | Error> {
    try {
        const file = await fs.readFile(filename(id), 'utf8');
        return file;
    } catch (error: any) {
        return new Error(error.message);
    }
}

