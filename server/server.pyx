import torch
import torch.nn.functional as F
import json
import torch.nn as nn

from snake2d import Snake, GAME_BOUNDS, INITIAL_SCORE
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
app = FastAPI()

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
print([i.weight.shape for i in W if isinstance(i, nn.Linear)])

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
            print(len(N))    
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
            print(len(N))    
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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print('Client connected!')
    try:
        
        while True:
            data = await websocket.receive_text()
            print(json.loads(data))
            refresh()
            return_data = {
                "snake_data": [snake.__dict__() for snake in population]
            }
            # Echo the message back to the client
            await websocket.send_text(json.dumps(return_data))
    except Exception as e:
        print(f"Client disconnected: {e}")
        
@app.websocket("/start")
async def get():
    start()



