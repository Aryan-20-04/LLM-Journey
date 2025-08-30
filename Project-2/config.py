# =============================
# 🚗 Simulation Configuration
# =============================

# Screen & Timing
SCREEN_W, SCREEN_H = 800, 600   # Window size
FPS = 60                        # Frames per second
dt = 1.0 / FPS                  # Simulation step size

# =============================
# 🏁 Track Parameters
# =============================
TRACK_WIDTH = 60                # Width of the racing track
TRACK_MARGIN = 20               # Margin around the track
CHECKPOINT_RADIUS = 80          # Distance to count checkpoint as reached

# =============================
# 🚙 Car Parameters
# =============================
CAR_WIDTH = 8                   # Car sprite width
CAR_HEIGHT = 16                 # Car sprite height
MAX_SPEED = 8.0                 # Top speed of the car
ACCELERATION = 0.3              # Acceleration per update
FRICTION = 0.92                 # Speed decay factor (0–1)
TURN_SPEED = 3.0                # Steering speed
CRASH_PENALTY = -300            # Fitness penalty for crashing
SMOOTHNESS_WEIGHT = 0.2         # Weight for smooth driving penalty
EXPONENTIAL_REWARD_BASE = 1.2   # Growth factor for checkpoint rewards

# =============================
# 🧠 AI / Neural Net Parameters
# =============================
N_SENSORS = 12                  # Number of distance sensors
SENSOR_RANGE = 250              # Max range of sensors (pixels)
POPULATION_SIZE = 50            # Number of agents in each generation
INITIAL_MUTATION_RATE = 0.25    # Starting mutation probability
MIN_MUTATION_RATE = 0.05        # Minimum mutation probability
ELITE_COUNT = 6                 # Top agents carried unchanged to next gen
SPECIES_THRESHOLD = 0.3         # Similarity threshold for speciation
HIDDEN_SIZES = [32, 24]         # Neural network hidden layer sizes
NUM_CHECKPOINTS = 24            # Number of checkpoints on the track
# =============================
# 🎓 Training Parameters
# =============================
TRAINING_MODE = False            # True = training mode, False = demo best agent
LOAD_BEST_MODEL = None          # Path to .npz to preload best model (optional)

BASE_GEN_TIME = 25.0            # Base time per generation (seconds)
EXTRA_TIME_PER_GEN = 0.2        # Additional time added per generation
MAX_GEN_TIME = 60.0             # Upper limit for generation time
MAX_GENERATIONS = 150           # Total generations to train

# =============================
# 🔧 Improvements & Options
# =============================
SAVE_TOP_N = 3                  # Number of top models saved per generation
SMOOTH_DRIVING_PENALTY = True   # Apply penalty for jerky driving
DYNAMIC_MUTATION = True         # Adaptive mutation over time
ENABLE_SPECIATION = True        # Keep genetic diversity via species
HEADLESS = True                 # If True, run without graphics (faster)