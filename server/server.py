import torch
import torch.nn.functional as F
import asyncio
import signal
#from websockets.asyncio.server import serve
import json
import time
import torch.nn as nn
import random
import math

from snake2d import Snake, GAME_BOUNDS, INITIAL_SCORE
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse



if torch.cuda.is_available():
    torch.cuda.set_device(0)

INPUT_LAYER_SIZE = 6
OUTPUT_LAYER_SIZE = 3
HIDDEN_LAYER_SIZE = 12

n_snakes = 20
population = [Snake(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE) for _ in range(n_snakes)]
food_arr = [torch.randint(-GAME_BOUNDS, GAME_BOUNDS, (2,)).float()] 
alpha = 0.0001
sigma = 1
best = 0
W = population[0].nn.ffwd

N = [nn.Sequential(
        nn.Linear(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, bias=False),
        nn.ReLU(),
        nn.Linear(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, bias=False),
        nn.ReLU()
    ) for i in range(n_snakes)]
       
def reset(n):
    global n_snakes, population, food_arr, alpha, sigma, best
    n_snakes = n
    population = [Snake(INPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE) for _ in range(n_snakes)]
    food_arr = [torch.randint(-GAME_BOUNDS, GAME_BOUNDS, (2,)).float()] 
    alpha = 0.0001
    sigma = 1
    best = 0
    W = population[0].nn.ffwd

def evaluate():
    global W
    R = torch.zeros(n_snakes)
    for i, snake in enumerate(population):
        R[i] = snake.fitness
    
    # Normalize the fitness scores
    A = (R - torch.mean(R)) / (torch.std(R))
    
    if torch.mean(R) > best:   
        for i, W_param in enumerate(W):
            if isinstance(W_param, nn.ReLU):
                continue
            X = torch.stack([N[n][i].weight for n in range(n_snakes)])
            W[i].weight = nn.Parameter((W[i].weight.T + alpha/(n_snakes*sigma) * (X.T @ A)).T)
            
    start()

def refresh():
    if not any([s.alive for s in population]):
        evaluate()
    
    for i, snake in enumerate(population):
        while population[i].score - INITIAL_SCORE >= len(food_arr):
            rand_food = torch.randint(-GAME_BOUNDS, GAME_BOUNDS, (2,)).float()
            food_arr.append(rand_food)
            
        if not population[i].alive:
            continue
        food = food_arr[population[i].score - INITIAL_SCORE]
        inputs = population[i].get_inputs(food)
        logits = population[i].nn(inputs)
        population[i].step(int(torch.argmax(logits).item()) - 1, food)
        
def start():
    global food_arr, N, population
    if len(food_arr) == 1:
        food_arr = [torch.randint(-GAME_BOUNDS, GAME_BOUNDS, (2,)).float()]
    else:
        food_arr = [food_arr[-1]]
    
    N = [nn.Sequential(
        nn.Linear(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, bias=False),
        nn.ReLU(),
        nn.Linear(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, bias=False),
        nn.ReLU()
    ) for i in range(n_snakes)]
    for j in range(n_snakes):
        population[j].setup()
        
        for i, W_param in enumerate(N[j]):
            if isinstance(W_param, nn.ReLU):
                continue
            
            population[j].nn.ffwd[i].weight = nn.Parameter(W_param.weight + sigma*N[j][i].weight)



def main():
    global food_arr, W, N
    if len(food_arr) == 1:
        food_arr = [torch.randint(-GAME_BOUNDS, GAME_BOUNDS, (2,)).float()]
    else:
        food_arr = [food_arr[-1]]
    
    N = [nn.Sequential(
        nn.Linear(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, bias=False),
        nn.ReLU(),
        nn.Linear(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, bias=False),
        nn.ReLU()
    ) for i in range(n_snakes)]
    for j in range(n_snakes):
        population[j].setup()
        
        for i, W_param in enumerate(N[j]):
            if isinstance(W_param, nn.ReLU):
                continue
               
            population[j].nn.ffwd[i].weight = nn.Parameter(W_param.weight + sigma*N[j][i].weight)
    
    
    while any([s.alive for s in population]):
        for i, snake in enumerate(population):
            while population[i].score - INITIAL_SCORE >= len(food_arr):
                rand_food = torch.randint(-GAME_BOUNDS, GAME_BOUNDS, (2,)).float()
                food_arr.append(rand_food)
                
            if not population[i].alive:
                continue
            food = food_arr[population[i].score - INITIAL_SCORE]
            inputs = population[i].get_inputs(food)
            logits = population[i].nn(inputs)
            population[i].step(int(torch.argmax(logits).item()) - 1, food)
    
    R = torch.zeros(n_snakes)
    for i, snake in enumerate(population):
        R[i] = snake.fitness
    
    # Normalize the fitness scores
    A = (R - torch.mean(R)) / (torch.std(R))
    
    if torch.mean(R) > best:   
        for i, W_param in enumerate(W):
            if isinstance(W_param, nn.ReLU):
                continue
            X = torch.stack([N[n][i].weight for n in range(n_snakes)])
            W[i].weight = nn.Parameter((W[i].weight.T + alpha/(n_snakes*sigma) * (X.T @ A)).T)

# uvicorn server:app --host 0.0.0.0 --port 8000 --reload
        
import asyncio
import json
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from signal import SIGINT, SIGTERM
import signal

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1"],  # Restrict access to NextJS
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Store active clients
active_clients = set()

# Dummy data generator
async def main_endpoint(client_id: str):
    """Async generator for data streaming."""
    try:
        while True:
            # This would be your `refresh()` logic
            await asyncio.sleep(0.01)  # Simulate a delay for data refresh
            refresh()
            data = {
                "id": client_id,
                "snake_data": [snake.__dict__() for snake in population]  # Example data
            }
            yield f"data: {json.dumps(data)}\n\n"
    except asyncio.CancelledError:
        print(f"Stream for client {client_id} was cancelled.")
    finally:
        # Clean up client when the stream ends
        if client_id in active_clients:
            active_clients.remove(client_id)
            print(f"Client {client_id} disconnected. Total clients: {len(active_clients)}")


@app.get("/stream")
async def stream(request: Request):
    """Streaming response for multiple clients."""
    client_id = str(id(request))  # Unique client identifier
    active_clients.add(client_id)
    print(f"New client connected: {client_id}. Total clients: {len(active_clients)}")

    # Track client disconnect
    async def client_disconnect():
        while True:
            if await request.is_disconnected():
                print(f"Client {client_id} disconnected.")
                active_clients.remove(client_id)
                break
            await asyncio.sleep(1)  # Check every second

    asyncio.create_task(client_disconnect())

    return StreamingResponse(main_endpoint(client_id), media_type="text/event-stream")


# Graceful shutdown logic
@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down server. Closing all client connections...")
    for client in list(active_clients):
        active_clients.remove(client)
    print("All clients have been disconnected.")