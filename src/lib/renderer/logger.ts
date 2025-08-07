export default class Logging {
    log(...message: any[]) {
        this.emit("info", ...message);
    }
    
    warn(...message: any[]) {
        this.emit("warn", ...message);
    }
    
    error(...message: any[]) {
        this.emit("error", ...message);
    }
    
    emit(messageType: "debug" | "warn" | "info" | "error", ...message: any[]) {
        console[messageType](`[${this.constructor.name}]:`, ...message);
    }
}