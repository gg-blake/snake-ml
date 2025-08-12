# Snake-ML: Machine Learning for N-Dimensional Spatial Reasoning

[![Live Demo](https://img.shields.io/badge/Demo-Live-green)](https://snakeml.moody.mx/)
[![Paper](https://img.shields.io/badge/Paper-Available-blue)](#citation)
[![License](https://img.shields.io/badge/License-MIT-yellow)](#license)

**Snake-ML** is a web-based simulation tool designed as an efficient and intuitive test bed for developing spatial navigation strategies in n-dimensional environments. Built entirely for the web using WebGL and TensorFlow.js, it enables privacy-preserving machine learning training directly on edge devices.

## 🌟 Features

- **N-Dimensional Snake Game**: Generalized classic snake game to any number of dimensions (N ≥ 2)
- **Edge Computing**: Train models directly in your browser with no cloud dependency
- **Genetic Algorithm**: Computationally efficient training without expensive backpropagation
- **Real-time Visualization**: WebGL-powered 3D visualization of the training process
- **Advanced Computer Vision**: Custom algorithms for boundary detection and collision avoidance
- **Unified WebGL Context (UWC)**: GPU pipeline achieving up to 32× speedup in training time
- **Data Augmentation**: Equivariant preprocessing for improved model efficiency

## 🚀 Quick Start

### Online Demo
Visit [https://snakeml.moody.mx/](https://snakeml.moody.mx/) to try Snake-ML immediately in your browser.

### Local Development

```bash
# Clone the repository
git clone https://github.com/gg-blake/snake-ml.git
cd snake-ml

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

## 🧠 How It Works

### Architecture Overview

Snake-ML uses a genetic learning algorithm to train neural networks that control n-dimensional snakes. Each snake is represented by a simple feedforward neural network with two layers:

| Layer | Input Shape | Output Shape | Activation | Parameters |
|-------|-------------|--------------|------------|------------|
| FC1   | (B, 3N-1)   | (B, 24)      | Tanh       | 48B(2N-1)  |
| FC2   | (B, 24)     | (B, N-1)     | Tanh       | -          |

Where:
- `B` = Batch size (number of snakes)
- `N` = Number of spatial dimensions

### Training Process

1. **Initialize Population**: Spawn B snakes with random neural network weights
2. **Parallel Simulation**: All snakes play simultaneously until failure
3. **Fitness Evaluation**: Rate snakes based on their performance using:
   ```
   score = ||p_prev - p_food|| - ||p_current - p_food||
   ```
4. **Selection & Reproduction**: Best-performing snakes breed to create the next generation
5. **Mutation**: Apply random mutations to encourage exploration

### Data Augmentation

Each snake receives augmented environmental data:

- **Wall Distance**: Custom raycasting algorithm calculates distance to boundaries
- **Food Direction**: Decomposed angles between snake direction and food position
- **Body Collision**: Proximity detection for self-collision avoidance

## 🔧 Configuration

### Basic Parameters

```typescript
const modelConfig: Config = {
    stepSize: 1,
    ttl: 200, // Number of steps snake is permitted without acquiring food
    batchInputShape: [
        100, // Number of snakes per generation
        10, // Capped length of snakes
        3 // Number of spatial dimensions (N ≥ 2)
    ],
    startingLength: 5,
    boundingBoxLength: 30, // Half-length of bounding box
    units: 24,
    fitnessGraphParams: {
        a: 10,
        b: 1.5,
        c: 4,
        min: -1,
        max: 1,
    },
    dtype: "float32",
};
const trainingConfig: TrainingConfig = {
    mutationFactor: 0.1, // Maximum mutation magnitude
    mutationRate: 0, // Probability of parameter mutation
};
```

### Advanced Settings

- **Genetic Algorithm**: Customize selection pressure, crossover methods
- **Rendering**: Toggle visualization, adjust frame rates, camera controls
- **Performance**: Enable/disable Unified WebGL Context for maximum performance

## 📊 Performance

### Benchmarks (AMD Ryzen 9 5900HS + RTX 3070)

| Configuration | Cloud Training | Edge Training | Edge + UWC | Speedup |
|---------------|----------------|---------------|------------|---------|
| 300 snakes    | 430ms         | 107ms         | 75ms       | 5.72×   |
| 1000 snakes   | 1447ms        | 335ms         | 83ms       | 17.53×  |
| 1900 snakes   | 2766ms        | 617ms         | 87ms       | 31.97×  |

**Key Findings:**
- Average 4.03× speedup over cloud computing
- Up to 31.97× speedup with Unified WebGL Context
- Performance scales with population size

## 🔬 Research Applications

Snake-ML is designed for research in:

- **Spatial Cognition**: Understanding how agents navigate complex environments
- **Edge AI**: Privacy-preserving machine learning on client devices
- **Computer Vision**: Real-time visual processing for navigation
- **Robotics**: Spatial reasoning for autonomous systems

## 🛠️ Technical Details

### Dependencies

- **TensorFlow.js**: GPU-accelerated machine learning in the browser
- **Three.js**: WebGL-based 3D visualization
- **WebGL**: Hardware-accelerated graphics rendering
- **Web Workers**: Parallel processing for training and rendering

### Browser Requirements

- Modern browser with WebGL 1.0+ support
- GPU recommended for optimal performance
- JavaScript enabled

### System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Browser   │    │  TensorFlow.js  │    │     Three.js    │
│                 │    │   (Training)    │    │ (Visualization) │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│     WebGL       │◄──►│  Genetic Algo   │◄──►│  3D Rendering   │
│   (Unified)     │    │   Population    │    │   Real-time     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Long Term  
- [ ] Peer-to-peer federated learning via Bluetooth
- [ ] Integration with robotics simulators
- [ ] Real-world deployment tools for autonomous systems

## 📄 Citation

If you use Snake-ML in your research, please cite our paper:

```bibtex
@article{moody2025snakeml,
  title={Machine Learning for N-Dimensional Spatial Reasoning Tasks on the Web},
  author={Moody, Blake and Kim, JieHyun and Kim, Sanghyuk and Haehn, Daniel},
  journal={Frontiers in Computer Science},
  year={2025},
  publisher={University of Massachusetts Boston}
}
```

## 📞 Contact

- **Blake Moody** - [blake.moody001@umb.edu](mailto:blake.moody001@umb.edu)
- **Research Lab** - Machine Psychology, University of Massachusetts Boston
- **Demo** - [https://snakeml.moody.mx/](https://snakeml.moody.mx/)

## 📋 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- University of Massachusetts Boston Department of Computer Science
- WebGL and TensorFlow.js communities
- OpenAI Gym for inspiration
- Contributors and researchers using Snake-ML

---

**Keywords**: machine learning, genetic algorithm, edge computing, spatial reasoning, artificial intelligence, computer vision, n-dimensional navigation, WebGL, TensorFlow.js