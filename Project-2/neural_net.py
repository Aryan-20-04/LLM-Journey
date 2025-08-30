from typing import List,Optional
import math,csv,glob,time,pickle,random
import numpy as np
from config import *
import pygame
from track import Track
from car import Car

class NeuralNetwork:
    def __init__(self, input_size: int, hidden_sizes: Optional[List[int]] = None, output_size: int = 2):
        self.input_size = input_size
        self.hidden_sizes : List[int] = hidden_sizes or HIDDEN_SIZES
        self.output_size = output_size
        
        # Build network architecture
        layer_sizes = [input_size] + self.hidden_sizes + [output_size]
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization
            weight_scale = math.sqrt(2.0 / layer_sizes[i])
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * weight_scale
            b = np.random.randn(layer_sizes[i + 1]) * 0.05
            
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, inputs: List[float]) -> List[float]:
        """Forward propagation with improved activations"""
        x = np.array(inputs, dtype=np.float32)
        
        # Forward through hidden layers
        for i in range(len(self.weights) - 1):
            x = np.dot(x, self.weights[i]) + self.biases[i]
            x = np.maximum(0.1 * x, x)  # Leaky ReLU
        
        # Output layer with tanh
        x = np.dot(x, self.weights[-1]) + self.biases[-1]
        output = np.tanh(x)
        
        return output.tolist()
    
    def get_weights(self) -> List[np.ndarray]:
        """Get all weights and biases"""
        result = []
        for w, b in zip(self.weights, self.biases):
            result.append(w.copy())
            result.append(b.copy())
        return result
    
    def set_weights(self, weights: List[np.ndarray]):
        """Set weights and biases from list"""
        self.weights = []
        self.biases = []
        for i in range(0, len(weights), 2):
            self.weights.append(weights[i])
            self.biases.append(weights[i + 1])
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.15):
        """Mutate network parameters"""
        for weight_matrix in self.weights:
            mask = np.random.random(weight_matrix.shape) < mutation_rate
            weight_matrix[mask] += np.random.randn(np.sum(mask)) * mutation_strength
            
        for bias_vector in self.biases:
            mask = np.random.random(bias_vector.shape) < mutation_rate
            bias_vector[mask] += np.random.randn(np.sum(mask)) * mutation_strength
    
    def crossover(self, other: 'NeuralNetwork') -> 'NeuralNetwork':
        """Create offspring through crossover"""
        child = NeuralNetwork(self.input_size, self.hidden_sizes, self.output_size)
        
        my_weights = self.get_weights()
        other_weights = other.get_weights()
        child_weights = []
        
        for w1, w2 in zip(my_weights, other_weights):
            # Uniform crossover
            mask = np.random.random(w1.shape) < 0.5
            child_weight = np.where(mask, w1, w2)
            child_weights.append(child_weight)
        
        child.set_weights(child_weights)
        return child
    
    def calculate_distance(self, other: 'NeuralNetwork') -> float:
        """Calculate genetic distance for speciation"""
        distance = 0.0
        my_weights = self.get_weights()
        other_weights = other.get_weights()
        
        total_params = 0
        for w1, w2 in zip(my_weights, other_weights):
            distance += np.sum(np.abs(w1 - w2))
            total_params += w1.size
        
        return distance / total_params if total_params > 0 else 0


class Species:
    def __init__(self, representative: 'Agent'):
        self.representative = representative
        self.members = [representative]
        self.fitness_history = []
        self.stagnant_generations = 0
        
    def add_member(self, agent: 'Agent'):
        self.members.append(agent)
    
    def update_fitness(self):
        if self.members:
            avg_fitness = sum(agent.fitness for agent in self.members) / len(self.members)
            
            if self.fitness_history and avg_fitness <= max(self.fitness_history[-5:], default=0):
                self.stagnant_generations += 1
            else:
                self.stagnant_generations = 0
                
            self.fitness_history.append(avg_fitness)
    
    def is_compatible(self, agent: 'Agent', threshold: float) -> bool:
        distance = self.representative.network.calculate_distance(agent.network)
        return distance < threshold


class Agent:
    def __init__(self, track: Track):
        self.track = track
        self.car = Car(track)
        self.network = NeuralNetwork(N_SENSORS + 6, HIDDEN_SIZES, 2)
        self.fitness = 0
        
        self.species_id: Optional[Species] = None
        self.saved_generation: Optional[int] = None
        self.saved_fitness: Optional[float] = None
        self.saved_seed: Optional[int] = None

    def reset(self):
        """Reset the agent"""
        self.car.reset()
        self.fitness = 0

    def act(self, dt: float):
        """Get action from neural network"""
        if not self.car.alive:
            return
        
        if len(self.car.sensor_readings) != N_SENSORS + 6:
            return
            
        inputs = self.car.sensor_readings
        outputs = self.network.forward(inputs)
        
        # IMPROVED: Better output processing
        throttle = outputs[0]
        steering = outputs[1]
        
        # Smoother throttle control
        if abs(throttle) > 0.1:
            throttle = np.sign(throttle) * (abs(throttle)**1.5)
        else:
            throttle = 0.0  # Dead zone for coasting
        
        # Adaptive smoothing
        speed_factor = max(0.3, 1.0 - (self.car.speed / MAX_SPEED) * 0.7)
        steering = max(-1.0, min(1.0, steering * speed_factor))  # Smoother steering
        
        if TRAINING_MODE:
            throttle += np.random.normal(0, 0.05)
            steering += np.random.normal(0, 0.02)
            throttle = max(-1.0, min(1.0, throttle))
            steering = max(-1.0, min(1.0, steering))

        self.car.update(dt, throttle, steering)

    def evaluate(self) -> float:
        """Evaluate and return fitness"""
        self.fitness = self.car.get_fitness()
        return self.fitness

class Population:
    def __init__(self, size: int, track: Track):
        self.size = size
        self.track = track
        self.agents = [Agent(track) for _ in range(size)]
        self.generation = 0
        self.best_fitness = 0
        self.avg_fitness = 0
        self.best_generation=0
        self.mutation_rate= INITIAL_MUTATION_RATE
        # Enhanced tracking
        self.fitness_history = []
        self.best_agents_history = []
        self.species = []
        
        # Performance logging
        self.performance_log = []
        
    def evaluate_generation(self, max_time: float = 25.0):
        """IMPROVED: Better generation evaluation with early success detection"""
        # Reset all agents
        for agent in self.agents:
            agent.reset()
        
        start_time = time.time()
        frame_dt = 0.016

        print(f"\n=== Generation {self.generation} ===")
        
        # Track best performance during generation
        best_progress_so_far = 0
        last_progress_update = start_time
        
        # Run simulation with proper event handling
        frame_count = 0
        while time.time() - start_time < max_time:
            # Handle pygame events every few frames to keep window responsive
            if frame_count % 3 == 0:  # Every 3rd frame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        print("  Training interrupted by user")
                        return
            
            alive_count = 0
            current_best_progress = 0
            
            for agent in self.agents:
                if agent.car.alive:
                    agent.car.update_sensors()
                    agent.act(frame_dt)
                    alive_count += 1
                    current_best_progress = max(current_best_progress, agent.car.checkpoint_progress)
            
            # Check for significant progress
            if current_best_progress > best_progress_so_far:
                best_progress_so_far = current_best_progress
                last_progress_update = time.time()
                print(f"  Progress update: {current_best_progress} checkpoints reached!")
            
            # Early completion check
            if best_progress_so_far >= len(self.track.checkpoints):
                print(f"  ✓ Track completed! Ending generation early.")
                break
            max_gen_time=min(MAX_GEN_TIME,BASE_GEN_TIME+self.generation*EXTRA_TIME_PER_GEN)
            # End early if no progress for too long
            if time.time() - last_progress_update > max_gen_time and alive_count < self.size // 4:
                print(f"  No progress for {max_gen_time:.1f}s with few survivors. Ending early.")
                break
                
            # End if no agents alive
            if alive_count == 0:
                print("  All agents died.")
                break
            
            frame_count += 1
        
        # Evaluate fitness for all agents
        fitness_scores = []
        max_checkpoints = 0
        max_forward = 0
        
        for agent in self.agents:
            fitness = agent.evaluate()
            fitness_scores.append(fitness)
            max_checkpoints = max(max_checkpoints, agent.car.checkpoint_progress)
            max_forward = max(max_forward, agent.car.total_forward_distance)
        
        # Update statistics
        self.best_fitness = max(fitness_scores) if fitness_scores else 0
        self.avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0
        self.fitness_history.append((self.best_fitness, self.avg_fitness))
        
        # Save top agents
        sorted_agents = sorted(self.agents, key=lambda a: a.fitness, reverse=True)
        self.best_agents_history.append([
            (agent.network.get_weights(), agent.fitness) 
            for agent in sorted_agents[:SAVE_TOP_N]
        ])
        
        # Log performance
        best_agent = sorted_agents[0] if sorted_agents else self.agents[0]
        self.performance_log.append({
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'avg_fitness': self.avg_fitness,
            'alive_count': sum(1 for a in self.agents if a.car.alive),
            'top_distance': best_agent.car.distance_traveled,
            'top_checkpoints': best_agent.car.checkpoint_progress,
            'top_forward_distance': best_agent.car.total_forward_distance,
            'unique_checkpoints': len(best_agent.car.checkpoint_visits)
        })
        
        print(f"Results: Best={self.best_fitness:.1f}, Avg={self.avg_fitness:.1f}")
        print(f"         Max checkpoints={max_checkpoints}, Max forward={max_forward:.1f}")
        
        # Show top 3 performances
        for i in range(min(3, len(sorted_agents))):
            agent = sorted_agents[i]
            unique = len(agent.car.checkpoint_visits)
            print(f"  #{i+1}: Fitness={agent.fitness:.1f}, "
                  f"Checkpoints={agent.car.checkpoint_progress}, "
                  f"Unique={unique}, Forward={agent.car.total_forward_distance:.1f}")
    
    def speciate(self):
        """Divide population into species"""
        if not ENABLE_SPECIATION:
            return
            
        # Clear existing species assignments
        for species in self.species:
            species.members.clear()
        
        # Assign agents to species
        for agent in self.agents:
            assigned = False
            for species in self.species:
                if species.is_compatible(agent, SPECIES_THRESHOLD):
                    species.add_member(agent)
                    agent.species_id = species
                    assigned = True
                    break
            
            if not assigned:
                new_species = Species(agent)
                agent.species_id = new_species
                self.species.append(new_species)
        
        # Remove empty species and update fitness
        self.species = [s for s in self.species if s.members]
        for species in self.species:
            species.update_fitness()
    
    def evolve(self):
        """IMPROVED: Better evolution strategy"""
        # Sort agents by fitness
        self.agents.sort(key=lambda a: a.fitness, reverse=True)
        
        # Apply speciation
        self.speciate()
        
        # Calculate dynamic mutation rate
        if DYNAMIC_MUTATION:
            progress = min(1.0, self.generation / MAX_GENERATIONS)
            # Start high, reduce as we progress, but keep minimum
            self.mutation_rate = INITIAL_MUTATION_RATE * (1 - progress * 0.6) + MIN_MUTATION_RATE
        else:
            self.mutation_rate = INITIAL_MUTATION_RATE

        new_agents = []
        
        # Keep elite agents (preserve best performers)
        elite_count = max(3, min(ELITE_COUNT, len(self.agents) // 8))
        for i in range(elite_count):
            new_agent = Agent(self.track)
            new_agent.network.set_weights(self.agents[i].network.get_weights())
            new_agents.append(new_agent)
        
        # Generate offspring with improved selection
        while len(new_agents) < self.size:
            # Select parents using tournament selection
            parent1 = self.tournament_selection(tournament_size=4)
            parent2 = self.tournament_selection(tournament_size=4)
            
            # Ensure different parents
            attempts = 0
            while parent2 == parent1 and attempts < 5:
                parent2 = self.tournament_selection(tournament_size=4)
                attempts += 1
            
            # Create offspring
            child_agent = Agent(self.track)
            
            # Crossover probability based on parent quality
            parent1_fitness = parent1.fitness
            parent2_fitness = parent2.fitness
            avg_parent_fitness = (parent1_fitness + parent2_fitness) / 2
            
            if avg_parent_fitness > self.avg_fitness and random.random() < 0.85:
                # High probability crossover for good parents
                child_agent.network = parent1.network.crossover(parent2.network)
            else:
                # Clone the better parent
                better_parent = parent1 if parent1_fitness > parent2_fitness else parent2
                child_agent.network.set_weights(better_parent.network.get_weights())
            
            # Mutate offspring with adaptive rate
            mutation_strength = 0.15 * (1 + random.random() * 0.5)  # Variable mutation strength
            child_agent.network.mutate(self.mutation_rate, mutation_strength)
            new_agents.append(child_agent)
        
        self.agents = new_agents
        self.generation += 1
        
        print(f"Evolution: Mutation rate={self.mutation_rate:.3f}, Elite={elite_count}")
    
    def tournament_selection(self, tournament_size: int = 4) -> Agent:
        """Improved tournament selection"""
        # Select from top 60% of population
        top_population = self.agents[:int(len(self.agents) * 0.6)]
        if not top_population:
            return random.choice(self.agents)
            
        tournament = random.sample(top_population, min(tournament_size, len(top_population)))
        return max(tournament, key=lambda a: a.fitness)
    
    def get_best_agent(self) -> Agent:
        """Get the best performing agent"""
        return max(self.agents, key=lambda a: a.fitness)
    
    def save_performance_log(self, filename: str = "training_log.csv"):
        """Save performance log to CSV"""
        if not self.performance_log:
            return    
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = self.performance_log[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.performance_log)
    
    def save_checkpoint(self, filename: str):
        """Save full population (agents, generation, fitness history, etc.)"""
        if filename is None:
            filename=f"checkpoint_gen_{self.generation}.pkl"
        
        state = {
            "generation": self.generation,
            "agents": [(a.network.get_weights(), a.fitness) for a in self.agents],
            "best_fitness": self.best_fitness,
            "avg_fitness": self.avg_fitness,
            "fitness_history": self.fitness_history,
            "mutation_rate": self.mutation_rate
        }
        with open(filename, "wb") as f:
            pickle.dump(state, f)
        print(f"💾 Population checkpoint saved: {filename}")

    @staticmethod
    def load_checkpoint(filename: str, track: Track) -> "Population":
        """Load population from checkpoint file"""
        try:
            with open(filename, "rb") as f:
                state = pickle.load(f)
            
            population = Population(len(state["agents"]), track)
            population.generation = state["generation"]
            population.best_fitness = state["best_fitness"]
            population.avg_fitness = state["avg_fitness"]
            population.fitness_history = state["fitness_history"]
            population.mutation_rate = state.get("mutation_rate", INITIAL_MUTATION_RATE)
            
            population.agents= []
            for weights,fitness in state["agents"]:
                agent=Agent(track)
                agent.network.set_weights(weights)
                agent.fitness=fitness
                population.agents.append(agent)
            
            print(f"✅ Population checkpoint loaded: {filename}")
            return population
        except Exception as e:
            print(f"❌ Error loading checkpoint from {filename}: {e}")
            return Population(POPULATION_SIZE, track)
