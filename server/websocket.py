#!/usr/bin/env python

import asyncio
import signal

from websockets.asyncio.server import serve

async def handle_control_channel(websocket):
    async for message in websocket:
        print(f"Received message (control): {message}")
        await websocket.send(f"Echo (control): {message}")

async def handle_data_channel(websocket):
    async for message in websocket:
        print(f"Received message (data): {message}")
        await websocket.send(f"Echo (data): {message}")

async def server():
    # Set the stop condition when receiving SIGTERM.
    loop = asyncio.get_running_loop()
    stop = loop.create_future()
    loop.add_signal_handler(signal.SIGTERM, stop.set_result, None)

    async with serve(handle_control_channel, "localhost", 6600):
        await stop

    async with serve(handle_data_channel, "localhost", 6601):
        await stop

asyncio.run(server())