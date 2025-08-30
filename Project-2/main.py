import time,pickle,sys,os,pygame,glob
import numpy as np
from typing import List, Tuple, Optional,Dict,Any
from config import *
from vector2 import Vector2
from track import Track
from car import Car
from neural_net import NeuralNetwork,Agent,Population,Species
from visualize import Visualizer

def load_for_demo(track: Track) -> Optional[Agent]:
    if os.path.exists("best_model.npz"):
        agent=load_best_model("best_model.npz", track)
        if agent:
            return agent
        print(f"❌ Failed to load model from best_model.npz")
        
    checkpoints=sorted(glob.glob("checkpoint_gen_*.npz"), key=lambda f: int(f.split('_')[2]))
    if checkpoints:
        latest=checkpoints[-1]
        agent=load_best_model(latest,track)
        if agent:
            print(f"👉 Using fallback checkpoint: {latest}")
            return agent
    print("❌ No saved models found. Please run training first.")
    return None

def load_best_model(filename: str, track: Track) -> Optional[Agent]:
    """Load a trained model from file"""
    try:
        data = np.load(filename, allow_pickle=True)
        agent = Agent(track)
        
        def key_index(k: str)->int:
            if k.startswith('arr_'):
                return int(k.split("_")[1])
            if k.startswith("w"):
                return int(k[1:])
            return 10**9

        weight_keys=[k for k in data.files if k.startswith("arr_")or k.startswith("w")]
        weight_keys.sort(key=key_index)
        
        if not weight_keys:
            raise ValueError("No weight data found in the file.")
        
        weights=[data[k] for k in weight_keys]
        agent.network.set_weights(weights)
        
        fitness=data["fitness"].item() if "fitness" in data.files else None
        generation=data["generation"].item() if "generation" in data.files else None
        
        if fitness is not None:
            agent.saved_fitness=float(fitness)
            agent.fitness=float(fitness)
            
        if generation is not None:
            agent.saved_generation=int(generation)
            
        print(f"✅ Loaded model from {filename}")
        if fitness is not None and generation is not None:
            print(f"   (fitness={fitness:.1f}, generation={generation})")
        return agent
    
    except Exception as e:
        print(f"Error loading model from {filename}: {e}")
        return None

def save_best_models(population: Population, generation: int):
    """Save top N models to files"""
    try:
        sorted_agents = sorted(population.agents, key=lambda a: a.fitness, reverse=True)
        best_agent=sorted_agents[0]
        
        weights=best_agent.network.get_weights()
        save_dict: Dict[str, Any] = {f"w{i}": w for i, w in enumerate(weights)}
        save_dict["fitness"] = np.array(float(best_agent.fitness))
        save_dict["generation"] = np.array(int(generation))
        np.savez("best_model.npz", **save_dict)
        
        np.savez(f"best_model_gen_{generation}.npz", **save_dict)
        

        # Optionally save a checkpoint every 25 generations
        if generation % 25 == 0 and generation > 0:
            with open(f"checkpoint_gen_{generation}.pkl", "wb") as f:
                pickle.dump({
                    "generation": generation,
                    "agents": [(a.network.get_weights(), a.fitness) for a in sorted_agents],
                    "best_fitness": population.best_fitness,
                    "avg_fitness": population.avg_fitness,
                    "fitness_history": population.fitness_history
                }, f)
            print(f"💾 Saved checkpoint for generation {generation}")

        print(f"✅ Best model updated (gen {generation}, fitness {best_agent.fitness:.1f})")
        return True
    except Exception as e:
        print(f"Error saving models: {e}")
        return False
    
def resume_or_new_pop(track :Track) -> Population:
    checkpoints=sorted(glob.glob("checkpoint_gen_*.pkl"))
    if checkpoints:
        latest= checkpoints[-1]
        print(f"👉 Resuming from checkpoint: {latest}")
        return Population.load_checkpoint(latest, track)
    else:
        print("✨ Starting fresh training population")
        return Population(POPULATION_SIZE, track)

def demo_mode(track: Track, model_path: Optional[str] = None):
    """Enhanced demo mode"""
    
    population = Population(1, track)
    visualizer = Visualizer()
    
    if model_path and os.path.exists(model_path):
        best_agent = load_best_model(model_path, track)
        if best_agent:
            population.agents = [best_agent]
            population.generation = getattr(best_agent, 'saved_generation', 0)
            population.best_fitness = getattr(best_agent, "saved_fitness", 0.0)
            print(f"✅ Loaded model from {model_path}")
            print(f"   (fitness={best_agent.saved_fitness:.1f}, generation={best_agent.saved_generation})")
            try:
                population.generation = int(model_path.split('_')[3])
            except:
                population.generation = 0
            population.best_fitness = getattr(best_agent, "saved_fitness", 0.0)
    
    print("IMPROVED Demo Mode: Showing best trained agent")
    print("Controls: ESC to quit, R to reset car")
    print("-" * 40)
    
    agent=population.agents[0]
    agent.car.update_sensors()
    stagnation_timer=0.0
    last_progress= agent.car.checkpoint_progress
    population.agents[0].car.update_sensors()
    try:
        while True:
            actions = visualizer.check_events()
            
            if actions['quit']:
                break
            if actions['reset'] or not agent.car.alive:
                agent.reset()
                agent.car.update_sensors()
                stagnation_timer=0.0
                last_progress=agent.car.checkpoint_progress
            
            dt = visualizer.clock.tick(60) / 1000.0
            # Update agent
            agent.act(dt)
            
            if not agent.car.alive:
                reason=getattr(agent.car, "crash_reason", "Unknown")
                print(f"❌ Agent crashed or died. Ending demo. Reason: {reason}")
                break
            
            if agent.car.checkpoint_progress > last_progress:
                stagnation_timer=0.0
                last_progress=agent.car.checkpoint_progress
            else:
                stagnation_timer+=dt
                if stagnation_timer > 12.0:
                    print("❌ Agent stagnated for too long. Ending demo.")
                    break
                
            # Render
            visualizer.render(track, population, show_sensors=True, training_mode=False)
            
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    finally:
        pygame.quit()
        print("Demo completed!")

def training_mode(track: Track):
    """IMPROVED: Enhanced training mode with proper visualization"""
    population = resume_or_new_pop(track)
    visualizer = Visualizer()

    best_agent_ever = None
    best_generation = 0

    print("Starting IMPROVED 2D Racing AI Training...")
    print(f"Population: {POPULATION_SIZE}")
    print(f"Max Generations: {MAX_GENERATIONS}")
    print(f"Track: {track.track_type} with {len(track.checkpoints)} checkpoints")
    print(f"Improvements: Better checkpoints, direction sensor, adaptive evolution")
    print("Controls: ESC to quit and show best result")
    print("-" * 60)

    last_render_time = time.time()
    render_interval = 1.0 / 30

    try:
        while population.generation < MAX_GENERATIONS:
            print(f"\n=== Generation {population.generation} ===")
            
            # Reset all agents for this generation
            for agent in population.agents:
                agent.reset()
            
            # Run generation with real-time visualization
            start_time = time.time()
            best_progress_so_far = 0
            last_progress_update = start_time
            
            max_gen_time=min(MAX_GEN_TIME,BASE_GEN_TIME+population.generation*EXTRA_TIME_PER_GEN)
            
            while time.time() - start_time < max_gen_time:
                current_time = time.time()
                
                if HEADLESS and current_time-last_render_time<render_interval:
                    time.sleep(0.001)
                    continue
        
                # Handle pygame events to keep window responsive
                actions = visualizer.check_events()
                if actions['quit']:
                    print("\nTraining interrupted by user")
                    # Calculate final fitness before returning
                    for agent in population.agents:
                        agent.evaluate()
                    return population
                
                # Update simulation
                dt = visualizer.clock.tick(60) / 1000.0
                alive_count = 0
                current_best_progress = 0
                
                for agent in population.agents:
                    if agent.car.alive:
                        agent.act(dt)
                        alive_count += 1
                        current_best_progress = max(current_best_progress, agent.car.checkpoint_progress)
                
                # Track progress
                if current_best_progress > best_progress_so_far:
                    best_progress_so_far = current_best_progress
                    last_progress_update = time.time()
                    print(f"  Progress: {current_best_progress} checkpoints!")
                
                # Render the current state
                best_agent = max(population.agents, key=lambda a: a.car.checkpoint_progress)
                
                if not HEADLESS or current_time - last_render_time >= render_interval:
                    show_sensors = best_agent.car.checkpoint_progress > 0
                    visualizer.render(track, population, show_sensors=show_sensors, training_mode=True)
                    last_render_time=current_time
                # Early termination conditions
                if best_progress_so_far >= len(track.checkpoints):
                    print(f"  ✓ Track completed! Ending generation early.")
                    break
                
                if time.time() - last_progress_update > max_gen_time/2 and alive_count < population.size // 4:
                    print(f"  Stagnant, ending early.")
                    break
                    
                if alive_count == 0:
                    print("  All agents died.")
                    break
            
            # Evaluate fitness for all agents after simulation
            fitness_scores = []
            max_checkpoints = 0
            max_forward = 0
            
            for agent in population.agents:
                fitness = agent.evaluate()
                fitness_scores.append(fitness)
                max_checkpoints = max(max_checkpoints, agent.car.checkpoint_progress)
                max_forward = max(max_forward, agent.car.total_forward_distance)
            
            # Update population statistics
            population.best_fitness = max(fitness_scores) if fitness_scores else 0
            population.avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0
            population.fitness_history.append((population.best_fitness, population.avg_fitness))
            
            # Save top agents
            sorted_agents = sorted(population.agents, key=lambda a: a.fitness, reverse=True)
            population.best_agents_history.append([
                (agent.network.get_weights(), agent.fitness) 
                for agent in sorted_agents[:SAVE_TOP_N]
            ])
            
            # Log performance
            best_agent = sorted_agents[0] if sorted_agents else population.agents[0]
            population.performance_log.append({
                'generation': population.generation,
                'best_fitness': population.best_fitness,
                'avg_fitness': population.avg_fitness,
                'alive_count': sum(1 for a in population.agents if a.car.alive),
                'top_distance': best_agent.car.distance_traveled,
                'top_checkpoints': best_agent.car.checkpoint_progress,
                'top_forward_distance': best_agent.car.total_forward_distance,
                'unique_checkpoints': len(best_agent.car.checkpoint_visits)
            })
            
            # Diversity measurement
            diversity = np.mean([
                np.linalg.norm(a.network.get_weights()[0] - best_agent.network.get_weights()[0])
               for a in population.agents
            ])
            print(f"   Diversity score (avg L2 distance): {diversity:.2f}")
            print(f"Results: Best={population.best_fitness:.1f}, Avg={population.avg_fitness:.1f}")
            print(f"         Max checkpoints={max_checkpoints}, Max forward={max_forward:.1f}")
            
            # Show top 3 performances
            for i in range(min(3, len(sorted_agents))):
                agent = sorted_agents[i]
                unique = len(agent.car.checkpoint_visits)
                print(f"  #{i+1}: Fitness={agent.fitness:.1f}, "
                      f"Checkpoints={agent.car.checkpoint_progress}, "
                      f"Unique={unique}, Forward={agent.car.total_forward_distance:.1f}")
            
            if population.generation % 25 == 0 and population.generation > 0:
                save_best_models(population, population.generation)
                print(f"  💾 Saved top 3 models from generation {population.generation}")
                population.save_checkpoint(f"population_checkpoint_gen_{population.generation}.pkl")

            # Save models periodically
            if best_agent_ever is None or best_agent.fitness>best_agent_ever.fitness:
                best_agent_ever=best_agent
                best_generation=population.generation
                weights=best_agent.network.get_weights()
                save_dict:Dict[str,Any] = {f"w{i}": w for i, w in enumerate(weights)}
                save_dict["fitness"] = np.array(float(best_agent.fitness))
                save_dict["generation"] = np.array(int(best_generation))

                np.savez("best_model.npz", **save_dict)
                print(f"  New best model saved from generation {best_generation} with fitness {best_agent.fitness:.1f}")
            
            # Check for good progress
            if population.best_fitness > 3000:
                print("  Great progress! AI is learning to navigate efficiently!")
            elif population.best_fitness > 1500:
                print("  Good progress detected!")
            
            # Early stopping if we achieve great performance
            if best_agent.car.checkpoint_progress >= len(track.checkpoints) * 2:
                print(f"\nExcellent performance achieved! Agent completed multiple laps.")
                break
            
            # Evolve to next generation
            population.evolve()
        
        # Training completed
        print(f"\nTraining completed after {population.generation} generations!")
        if best_agent_ever is not None:
            weights=best_agent_ever.network.get_weights()
            save_dict: Dict[str, Any] = {f"w{i}": w for i, w in enumerate(weights)}
            save_dict["fitness"] = np.array(float(best_agent_ever.fitness))
            save_dict["generation"] = np.array(int(best_generation))
            np.savez("best_model.npz", **save_dict)
            print(f"🏆 Final best model saved from gen {best_generation} with fitness {best_agent_ever.fitness:.1f}")
        
        population.save_performance_log("improved_training_log.csv")
        
        return population
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if best_agent_ever is not None:
            weights = best_agent_ever.network.get_weights()
            save_dict: Dict[str, Any] = {f"w{i}": w for i, w in enumerate(weights)}
            save_dict["fitness"] = np.array(float(best_agent_ever.fitness))
            save_dict["generation"] = np.array(int(best_generation))
            np.savez("best_model.npz", **save_dict)
            print(f"💾 Saved interrupted best model from gen {best_generation} with fitness {best_agent_ever.fitness:.1f}")
        population.save_performance_log("improved_training_log.csv")
        return population
    
def validate_config():
    """Validate all configuration parameters"""
    errors = []
    
    # Screen validation
    if not (400 <= SCREEN_W <= 2000) or not (300 <= SCREEN_H <= 1500):
        errors.append(f"Invalid screen size: {SCREEN_W}x{SCREEN_H}")
    
    # AI parameters
    if not (2 <= N_SENSORS <= 32):
        errors.append(f"N_SENSORS must be 2-32, got {N_SENSORS}")
    
    if not (10 <= POPULATION_SIZE <= 1000):
        errors.append(f"POPULATION_SIZE must be 10-1000, got {POPULATION_SIZE}")
    
    # Rates and probabilities
    if not (0.0 <= INITIAL_MUTATION_RATE <= 1.0):
        errors.append(f"INITIAL_MUTATION_RATE must be 0.0-1.0, got {INITIAL_MUTATION_RATE}")
    
    if not (0.0 <= MIN_MUTATION_RATE <= INITIAL_MUTATION_RATE):
        errors.append(f"MIN_MUTATION_RATE invalid: {MIN_MUTATION_RATE}")
    
    # Network architecture
    if not HIDDEN_SIZES or any(size <= 0 for size in HIDDEN_SIZES):
        errors.append(f"Invalid HIDDEN_SIZES: {HIDDEN_SIZES}")
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"- {error}" for error in errors))


def main():
    """VAlidation of config file"""
    validate_config()
    """Enhanced main function"""
    # Parse command line arguments
    track_type = "oval"  # default
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--demo":
            model_path = sys.argv[2] if len(sys.argv) > 2 else None
            track_type = sys.argv[3] if len(sys.argv) > 3 else "oval"
            track = Track(track_type)
            demo_mode(track, model_path)
            return
        elif sys.argv[1] == "--track":
            track_type = sys.argv[2] if len(sys.argv) > 2 else "oval"
    
    # Create track
    track = Track(track_type)
    
    # Check if we should load an existing model for demo
    if LOAD_BEST_MODEL and os.path.exists(LOAD_BEST_MODEL):
        demo_mode(track, LOAD_BEST_MODEL)
        return
    
    # Run training mode
    if TRAINING_MODE:
        track=Track("oval",seed=42)
        final_population = training_mode(track)
        
        # After training, show the best result
        print("\nTraining finished!")
        print("Showing best trained agent...")
        
        # Save the final best models
        best_agent=final_population.get_best_agent()
        
        if best_agent is None:
            print("❌ No valid agent found for demo.")
            return
        
        best_agent.saved_generation = final_population.generation
        best_agent.saved_fitness = final_population.best_fitness
        best_agent.saved_seed = 42  # keep seed for later reuse

        # Demo the best agent
        best_agent = final_population.get_best_agent()
        print(f"Best agent stats:")
        print(f"  Fitness: {best_agent.fitness:.1f}")
        print(f"  Checkpoints reached: {best_agent.car.checkpoint_progress}")
        print(f"  Unique checkpoints: {len(best_agent.car.checkpoint_visits)}")
        print(f"  Forward distance: {best_agent.car.total_forward_distance:.1f}")
        
        # Create demo population
        demo_population = Population(1, track)
        demo_population.agents = [best_agent]
        demo_population.generation = final_population.generation
        demo_population.best_fitness = final_population.best_fitness
        
        # Demo mode
        visualizer = Visualizer()
        print("\nDemo Mode: Showing best trained agent")
        print("Controls: ESC to quit, R to reset")
        print("-" * 40)
        
        try:
            while True:
                actions = visualizer.check_events()
                
                if actions['quit']:
                    break
                
                if actions['reset'] or not demo_population.agents[0].car.alive:
                    demo_population.agents[0].reset()
                
                dt = visualizer.clock.tick(60) / 1000.0
                demo_population.agents[0].act(dt)
                visualizer.render(track, demo_population, show_sensors=True, training_mode=False)
                
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        
        finally:
            pygame.quit()
            print("Training and demo completed!")
            print("Check 'improved_training_log.csv' for detailed performance data")
    
    else:
        # Direct demo mode
        track = Track("oval", seed=42)
        demo_mode(track, model_path="best_model.npz")

if __name__ == "__main__":
    main()