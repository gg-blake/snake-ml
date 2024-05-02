# SnakeMachineLearning
Simulating and facilitating an automated bot to play snake, training a deep learning model with a genetic learning algorithm. The project is currently being developed in Mojo, a new language being developed to optimize AI performance with readable syntax similar to Python.

![image](https://github.com/gg-blake/SnakeGameMachineLearning/assets/32070003/acc1be1a-611c-4922-890b-dd2693d60d2b)

## Development Setup
### Requirements
**OS:** Ubuntu 22.04 x86-64 (native or WSL)
**Software:** Mojo SDK, Python 3.10, Docker (Optional)
**Python Dependencies:** pygame, torch, termcolor

### Docker Setup Guide
1. Pull the docker image to your local machine with `docker pull ghcr.io/gg-blake/snake-ml-dev:latest`
2. After running the container, clone the repo
3. To run the program, type `mojo main.mojo`
4. To build an executable, type `mojo build main.mojo` then to run type `./main`

### Pip Setup Guide
1. Install Python 3.10
2. Follow the instructions to install mojo on Modular's website
3. Set the libpython path with the command `export MOJO_PYTHON_LIBRARY=$(modular config mojo.python_lib)`
4. Clone the repo
5. Install the python dependencies by issuing the command `pip install -r requirements.txt`, or pip installing pygame, termcolor, and torch individually
6. To run the program, type `mojo main.mojo`
7. To build an executable, type `mojo build main.mojo` then to run type `./main`
