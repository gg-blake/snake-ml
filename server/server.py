import torch
import torch.nn.functional as F
import asyncio
import signal
import json
import time
import torch.nn as nn
import random
import math

from nd_snake import NDSnake, INITIAL_SCORE
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse



if torch.cuda.is_available():
    torch.cuda.set_device(0)

N_DIMENSIONS = 2
DEGREES_OF_FREEDOM = (N_DIMENSIONS - 1) * N_DIMENSIONS
INPUT_LAYER_SIZE = N_DIMENSIONS * 6 - 2
HIDDEN_LAYER_SIZE = 20
OUTPUT_LAYER_SIZE = (( N_DIMENSIONS - 1 ) * N_DIMENSIONS) // 2
GAME_BOUNDS = 10
TTL = 100
TURN_ANGLE = 0
SPEED = 1

n_snakes = 10
population = [NDSnake(N_DIMENSIONS, HIDDEN_LAYER_SIZE, GAME_BOUNDS, SPEED, TTL, TURN_ANGLE) for _ in range(n_snakes)]
food_arr = [torch.randint(-GAME_BOUNDS, GAME_BOUNDS, (N_DIMENSIONS,)).float()] 
alpha = 0.001
sigma = 0.00001
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
    population = [NDSnake(N_DIMENSIONS, HIDDEN_LAYER_SIZE, GAME_BOUNDS, SPEED, TTL, TURN_ANGLE) for _ in range(n_snakes)]
    food_arr = [torch.randint(-GAME_BOUNDS, GAME_BOUNDS, (N_DIMENSIONS,)).float()] 
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
    global food_arr
    if not any([s.alive for s in population]):
        evaluate()
    
    for i, snake in enumerate(population):
        while population[i].score - INITIAL_SCORE >= len(food_arr):
            rand_food = torch.randint(-GAME_BOUNDS, GAME_BOUNDS, (N_DIMENSIONS,)).float()
            food_arr.append(rand_food)
            
        if not population[i].alive:
            continue
        food = food_arr[population[i].score - INITIAL_SCORE]
        inputs = population[i].get_inputs(food)
        
        logits = population[i].nn(inputs)
        
        if TURN_ANGLE > 0:
            logits = ((F.normalize(logits, dim=-1) - 0.5) * (torch.pi / 2))
            population[i].step(logits, food)
        else:
            logits = ((F.normalize(logits, dim=-1) - 0.5) * torch.pi)
            population[i].step(logits, food, discrete=True)
        
def start():
    global food_arr, N, population
    food_arr = [food_arr.pop()]
    
    N = [nn.Sequential(
        nn.Linear(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, bias=False),
        nn.ReLU(),
        nn.Linear(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, bias=False),
        nn.ReLU()
    ) for i in range(n_snakes)]
    for j in range(n_snakes):
        population[j].setup(GAME_BOUNDS, TURN_ANGLE, SPEED, TTL)
        
        for i, W_param in enumerate(N[j]):
            if isinstance(W_param, nn.ReLU):
                continue
            
            population[j].nn.ffwd[i].weight = nn.Parameter(W_param.weight + sigma*N[j][i].weight)

# uvicorn server:app --host 0.0.0.0 --port 8000 --reload
        
import asyncio
import json
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from signal import SIGINT, SIGTERM
import signal
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict access to NextJS
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Store active clients
active_clients = set()


async def main_endpoint(client_id: str):
    """Async generator for data streaming."""
    try:
        while True:
            # This would be your `refresh()` logic
            await asyncio.sleep(0.05)  # Simulate a delay for data refresh
            refresh()
            if len(food_arr) == 0:
                rand_food = torch.randint(-GAME_BOUNDS, GAME_BOUNDS, (N_DIMENSIONS,)).float()
                food_arr.append(rand_food)
            data = {
                "id": client_id,
                "snake_data": [snake.__dict__(i) for i, snake  in enumerate(population)],  # Example data
                "food_data": [f.tolist() for f in food_arr]
            }
            yield f"data: {json.dumps(data)}\n\n"
    except asyncio.CancelledError:
        print(f"Stream for client {client_id} was cancelled.")
    finally:
        # Clean up client when the stream ends
        if client_id in active_clients:
            active_clients.remove(client_id)
            print(f"Client {client_id} disconnected. Total clients: {len(active_clients)}")


class PostBody(BaseModel):
    n_dims: int
    ttl: int
    width: int
    turn_angle: float
    speed: float
    n_snakes: int
    sigma: float
    alpha: float
    hidden_layer_size: int

@app.post("/set")
async def set(body: PostBody):
    global N_DIMENSIONS, DEGREES_OF_FREEDOM, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, GAME_BOUNDS, TTL, TURN_ANGLE, population, alpha, sigma, food_arr, W, N, n_snakes
    N_DIMENSIONS = body.n_dims
    DEGREES_OF_FREEDOM = (N_DIMENSIONS - 1) * N_DIMENSIONS
    INPUT_LAYER_SIZE = N_DIMENSIONS * 6 - 2
    HIDDEN_LAYER_SIZE = body.hidden_layer_size
    OUTPUT_LAYER_SIZE = ((N_DIMENSIONS - 1) * N_DIMENSIONS) // 2
    GAME_BOUNDS = body.width
    TTL = body.ttl
    TURN_ANGLE = body.turn_angle
    SPEED = body.speed
    alpha = body.alpha
    sigma = body.sigma
    n_snakes = body.n_snakes
    population = [NDSnake(N_DIMENSIONS, HIDDEN_LAYER_SIZE, GAME_BOUNDS, SPEED, TTL, TURN_ANGLE) for _ in range(n_snakes)]
    food_arr = [torch.randint(-GAME_BOUNDS, GAME_BOUNDS, (N_DIMENSIONS,)).float()] 
    W = population[0].nn.ffwd

    N = [nn.Sequential(
        nn.Linear(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, bias=False),
        nn.ReLU(),
        nn.Linear(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, bias=False),
        nn.ReLU()
    ) for i in range(n_snakes)]

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