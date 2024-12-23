# Overview
Snake-ML is web-visualization of applying machine learning to a classic arcade game called snake. The premise is simple: navigate a space to gather food, avoid the walls of the game space, and avoid your growing tail. As the player progresses, navigating this space while avoiding the tail becomes increasingly difficult and requires thinking multiple steps ahead to avoid being trapped.
### Training an AI to Play
Training an AI to play a game is more difficult than simple classification of images or data as it falls under a category of *unsupervised learning*. This means that the AI must learn to play the game without any immediate training data and must learn through trial and error. Attempting to train an AI to play this game may seem trivial at first, but problems arise when we want the snake to do things the human player cannot (I will get into that soon).

The AI learns using a very famous unsupervised learning technique called a *genetic learning algorithm*. This technique was proposed in the early 1950s by none other than Alan Turing: the Father of Computing and was popularized in the late 1980s. This algorithm involves the simulating a large amount of parallel models, initially all randomly generated, then the models with the best calculated *fitness score* are then selected to carry on their unique attributes or *genes* to the next batch of simulated models. This directly mimics the evolutionary mechanics of the animal kingdom where survival of the fittest is law.

# Setup
#### Unfortunately, at this moment there is no way to run the model natively through a web client so to run the simulation, dependencies are required for your computer.

1) Fork the repo on the `web` branch and clone it to your local machine (download git [here](https://git-scm.com/downloads))
2) Install the latest Miniconda version for your computer [here](https://docs.anaconda.com/miniconda/install/)
3) Install node and npm [here](https://nodejs.org/en/download/package-manager)
4) In your terminal, navigate to the cloned repo directory and issue the following command to create the prebuilt conda environment
```
cd snake-ml
conda env create -f environment.yml
```
5) Once the environment is created, activate the environment with this command
```
conda activate snake-ml-env
```
6) Install the npm dependencies for the web server with the following commands
```
cd client
npm install --force
```
7) At this point, before opening the website, you will need two command prompts open to run the web server and the python server. I recommend using [tmux](https://github.com/tmux/tmux/wiki) so you don't have do this but its not required. Start the python server with the following commands:
```
cd server
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```
8) Lastly, you will start the web server with the following command
```
cd client
npm run dev
```
9) Initially, the web server is open locally on port 3000, you can access it at http://localhost:3000

# Usage
The snake simluation can be adjusted viausally and behaviorally in a side panel in the top-left of the screen. You can adjust the number of dimensions, game size, number of snakes and some other features like learning rate and a noise parameter.
![tweakpane-1.png](https://github.com/gg-blake/snake-ml/blob/web/tweakpane-1.png?raw=True)
In dimensions lower than 4, you can customize the color visualization. At the moment there are two vidualizations, `alive` (set by default) which highlights all snakes that are currently alive and `best` which highlights the curently best performing snake. I plan to add some more visualizations in the future so stayed tuned! Visualization mode can be access in the `Visual` tab
![tweakpane-2.png](https://github.com/gg-blake/snake-ml/blob/web/tweakpane-2.png?raw=True)
## Higher dimensional snake
The program currently has the ability to simulate an infinite number of dimensions for the snake to navigate, however only the lowest 4 dimensions can be visible on the screen. If you want to go beyond the fourth dimensions (just for fun) then you can adjust the presets manually in the `client/app/stream.tsx` file. The fourth dimension of the simulation is projected to the HSL color-space where lower `w` positions in the space correspond with a snake with a red hue and higher `w` positions correspond with a purple and blue hue.

I recommend observing the fact that the snakes in the fourth and higher dimensions seem to break the rules of the game when they moves forward and backwards and loop on each other. However, they are moving in a dimension that is inaccessible to humans. This gives insight into how 3-dimensional snakes appear in 2-dimensional cross sections. 
### 2-dimensions
![game-2d.png](https://github.com/gg-blake/snake-ml/blob/web/game-2d.gif?raw=True)
### 4-dimensions
![game-4d.png](https://github.com/gg-blake/snake-ml/blob/web/game-4d.gif?raw=True)
# Technologies
The backend relied on [PyTorch](https://pytorch.org/) for running and training the models. PyTorch is an open-source machine learning library for Python that is tuned for high-performance tensor operations for machine learning.

Additionally, the backend relied on [SSE](https://en.wikipedia.org/wiki/Server-sent_events), an alternative to [HTTP websockets](https://en.wikipedia.org/wiki/WebSocket) to send low-latency data stream to the web client. Low latency was crucial in this project as I wanted the user to see snake updates at a reasonable frame rate. However this came at a cost since SSE is similar to the [UDP layer 4 protocol](https://en.wikipedia.org/wiki/User_Datagram_Protocol) as it does not require the client to receive the data that is sent. However, since I am running this client server communication locally the frame loss can safely be ignored.

For the frontend, I am using a popular web framework called [Next.JS](https://nextjs.org/) which allows for real-time state changes in the [DOM](https://developer.mozilla.org/en-US/docs/Web/API/Document_Object_Model) to be partially re-rendered. This technology wasn't entirely necessary for my needs, however I am comfortable working with this framework for web development. I used this to reduce friction in the development process.
# Challenges Faced
### CORS
When attempting to communicate information using SSE from the server to client, the NextJS API was providing difficulty with establishing valid CORS headers. The most tricky part about this was the lack of information given as an error response from the server when the CORS headers were invalid.

![cors.png](https://github.com/gg-blake/snake-ml/blob/web/cors.png?raw=True)
### High-Dimensional Rotation
We learned in our CS 460 course that rotations can be applied to an object by applying what is called a rotation matrix to an object's vector. We learned about rotation in 2 and 3-dimensions but for the purposes of my project, I wanted to have a rotation matrix for every possible sized vector (3, 4, 5, 6... dimensions). According to [this paper](https://naos-be.zcu.cz/server/api/core/bitstreams/c155d250-c732-4256-a9cf-33cd61f0015f/content), a n-dimensional rotation can be represented as a series of 2-dimensional rotations along different 2-d subspaces of a higher dimensional space. Implementing this algorithm for my needs was extremely time-consuming. I needed to utilize this rotation algorithm so that each logit in the output nodes of the neural network when normalized, corresponds to a degree of rotation in radians.

# Acknowledgements
This project was in part developed with generative text models such as [GitHub Copilot](https://github.com/features/copilot) and [ChatGPT](https://chatgpt.com/)

For web development references I used the [ThreeJS API Docs](https://threejs.org/docs/) , [Mozilla Web Docs](https://developer.mozilla.org/en-US/), as well as [Stack Overflow](https://stackoverflow.com/)

Special thanks to Professor Daniel Haehn for his excellent CS 460 course. This project has been a long time in the making and his course has reinvigorated my passion for this project. I will continue to improve on the project and add more features.
