// src/app/api/hello.js
import { NextResponse, NextRequest } from 'next/server';

export async function GET() {
    return NextResponse.json({ message: 'Hello from API route!' });
}