🏎️ 2D Racing AI with Evolutionary Neural Networks

This project is a self-driving car simulator where agents learn to drive around a 2D track using evolutionary neural networks.
Cars sense the environment with virtual sensors, feed data into a configurable neural net, and improve over generations via mutation, crossover, and speciation.

✨ Features

Neuroevolutionary AI:

Adaptive mutation rate

Speciation to preserve diversity

Crossover + elitism

Car Physics & Fitness:

Virtual sensors detect walls and checkpoints

Fitness rewards for progress, smooth driving, and distance

Penalties for crashes and off-track driving

Visualization:

Real-time rendering with Pygame

Debug view: sensors, checkpoints, agent stats

Headless mode for faster training without graphics

Checkpointing & Resume:

Auto-save best models and full population snapshots

Resume training from checkpoints if available

📂 Project Structure
.
├── main.py           # Entry point (training + demo)
├── config.py         # Configuration & hyperparameters
├── neural_net.py     # Neural network, population, evolution logic
├── car.py            # Car physics, sensors, fitness
├── track.py          # Track generation & checkpoints
├── vector2.py        # Lightweight 2D vector math
├── visualize.py      # Visualization with pygame
└── checkpoints/      # Saved models and logs

🚀 Usage
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Quick Start
🔹 Fast Debug Training

For quick tests (less accurate but faster):

Set in config.py:

POPULATION_SIZE = 20
MAX_GENERATIONS = 30
HEADLESS = True


Run:

python main.py

🔹 Full Training

For best results (slower, more accurate):

Set in config.py:

POPULATION_SIZE = 50
MAX_GENERATIONS = 150
HEADLESS = False


Run:

python main.py

3️⃣ Resume Training

If a checkpoint (checkpoint.pkl or best_model.npz) exists, training will resume automatically (when enabled in config).

4️⃣ Demo Mode

Switch to demo mode in config.py:

TRAINING_MODE = False


Then run:

python main.py


This loads the best trained model and shows it driving on the track.

📊 Outputs

checkpoints/best_model.npz → best performing model weights

checkpoints/population.pkl → full population snapshot

improved_training_log.csv → training statistics per generation

🧪 Extending

Modify track.py to design new circuits.

Adjust reward shaping in car.py for different driving styles.

Experiment with network size & activations in neural_net.py.

Enable headless mode in config.py to speed up training.

📈 Roadmap

 Parallel training (multiprocessing)

 GUI config editor

 Replay system for best agents

🏆 Goal

The AI learns to navigate tracks efficiently, evolving from random driving to near-human racing behavior.