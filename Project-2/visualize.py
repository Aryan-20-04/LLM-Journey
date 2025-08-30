import pygame,math
from track import Track
from config import *
from car import Car
from neural_net import Population

class Visualizer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("2D Racing AI")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # For fitness graph
        self.fitness_surface = pygame.Surface((200, 100))
        
        
    def draw_track(self, track: Track):
        """Draw the racing track"""
        center = (SCREEN_W // 2, SCREEN_H // 2)
        
        if track.track_type == "oval":
            pygame.draw.ellipse(self.screen, (60, 60, 60), 
                              (center[0] - 320, center[1] - 210, 640, 420))
            pygame.draw.ellipse(self.screen, (40, 100, 40), 
                              (center[0] - 280, center[1] - 180, 560, 360))
            pygame.draw.ellipse(self.screen, (60, 60, 60), 
                              (center[0] - 140, center[1] - 90, 280, 180))
        else:
            # Generic track drawing for other types
            if len(track.walls) > 0:
                for wall in track.walls[:len(track.walls)//2]:
                    pygame.draw.line(self.screen, (60, 60, 60), 
                                   (wall[0].x, wall[0].y), (wall[1].x, wall[1].y), 3)
                for wall in track.walls[len(track.walls)//2:]:
                    pygame.draw.line(self.screen, (60, 60, 60), 
                                   (wall[0].x, wall[0].y), (wall[1].x, wall[1].y), 3)
        
        # Draw checkpoints with better visibility
        for i, checkpoint in enumerate(track.checkpoints):
            color = (255, 255, 0) if i == 0 else (255, 150, 100)
            pygame.draw.circle(self.screen, color, (int(checkpoint.x), int(checkpoint.y)), 10, 3)
            # Checkpoint numbers
            text = self.small_font.render(str(i), True, (255, 255, 255))
            self.screen.blit(text, (int(checkpoint.x) - 8, int(checkpoint.y) - 25))
            
    def draw_car(self, car: Car, color=(255, 255, 255)):
        """Draw a car with better visibility"""
        if not car.alive:
            color = (100, 100, 100)
            
        cos_a = math.cos(car.angle)
        sin_a = math.sin(car.angle)
        
        # Calculate car corners
        half_w, half_h = CAR_WIDTH // 2, CAR_HEIGHT // 2
        corners = [
            (-half_w, -half_h), (half_w, -half_h),
            (half_w, half_h), (-half_w, half_h)
        ]
        
        rotated_corners = []
        for x, y in corners:
            rx = x * cos_a - y * sin_a + car.pos.x
            ry = x * sin_a + y * cos_a + car.pos.y
            rotated_corners.append((rx, ry))
        
        pygame.draw.polygon(self.screen, color, rotated_corners)
        
        # Direction indicator (front of car)
        front_x = car.pos.x + cos_a * CAR_HEIGHT // 2
        front_y = car.pos.y + sin_a * CAR_HEIGHT // 2
        pygame.draw.circle(self.screen, (255, 100, 100), (int(front_x), int(front_y)), 4)
    
    def draw_sensors(self, car: Car):
        """Draw sensor rays"""
        if not car.alive:
            return
            
        for i in range(N_SENSORS):
            angle_range = math.pi * 0.8  # Match sensor range
            sensor_angle = car.angle + (i - N_SENSORS//2) * angle_range / (N_SENSORS - 1)
            distance = car.sensor_readings[i] * SENSOR_RANGE
            
            end_x = car.pos.x + math.cos(sensor_angle) * distance
            end_y = car.pos.y + math.sin(sensor_angle) * distance
            
            # Color based on distance
            intensity = int(255 * (1 - car.sensor_readings[i]))
            color = (255, 255 - intensity, 0)
            
            pygame.draw.line(self.screen, color, 
                           (car.pos.x, car.pos.y), (end_x, end_y), 2)
    
    def draw_progress_indicators(self, car: Car):
        """Draw progress indicators for the best car"""
        if not car.alive or not self.track.checkpoints:
            return
        
        # Draw line to next checkpoint
        next_checkpoint_idx = (car.last_checkpoint_index + 1) % len(self.track.checkpoints)
        next_checkpoint = self.track.checkpoints[next_checkpoint_idx]
        
        pygame.draw.line(self.screen, (255, 255, 0), 
                        (car.pos.x, car.pos.y), 
                        (next_checkpoint.x, next_checkpoint.y), 2)
        
        # Highlight visited checkpoints
        for visited_idx in car.checkpoint_visits:
            if visited_idx < len(self.track.checkpoints):
                checkpoint = self.track.checkpoints[visited_idx]
                pygame.draw.circle(self.screen, (0, 255, 0), 
                                 (int(checkpoint.x), int(checkpoint.y)), 12, 2)
    
    def draw_fitness_graph(self, population: Population):
        """Draw fitness evolution graph"""
        if len(population.fitness_history) < 2:
            return
            
        self.fitness_surface.fill((40, 40, 40))
        
        # Get data
        best_fitnesses = [f[0] for f in population.fitness_history]
        avg_fitnesses = [f[1] for f in population.fitness_history]
        
        if not best_fitnesses:
            return
            
        # Scale data to fit surface
        max_fitness = max(best_fitnesses) if best_fitnesses else 1
        if max_fitness == 0:
            max_fitness = 1
        width, height = self.fitness_surface.get_size()
        
        # Draw best fitness line
        if len(best_fitnesses) > 1:
            points = []
            for i, fitness in enumerate(best_fitnesses):
                x = int(i * width / max(1, len(best_fitnesses) - 1))
                y = int(height - (fitness / max_fitness) * height * 0.9)
                points.append((x, max(0, min(height-1, y))))
            
            if len(points) > 1:
                pygame.draw.lines(self.fitness_surface, (0, 255, 0), False, points, 2)
        
        # Draw average fitness line
        if len(avg_fitnesses) > 1:
            points = []
            for i, fitness in enumerate(avg_fitnesses):
                x = int(i * width / max(1, len(avg_fitnesses) - 1))
                y = int(height - (fitness / max_fitness) * height * 0.9)
                points.append((x, max(0, min(height-1, y))))
            
            if len(points) > 1:
                pygame.draw.lines(self.fitness_surface, (100, 150, 255), False, points, 2)
        
        # Blit to main screen
        self.screen.blit(self.fitness_surface, (SCREEN_W - 220, 20))
        
        # Labels
        best_text = self.small_font.render("Best", True, (0, 255, 0))
        avg_text = self.small_font.render("Avg", True, (100, 150, 255))
        self.screen.blit(best_text, (SCREEN_W - 220, 130))
        self.screen.blit(avg_text, (SCREEN_W - 180, 130))
    
    def draw_ui(self, population: Population, training_mode: bool = True):
        """Draw enhanced user interface"""
        y = 10
        if training_mode:
            species_count = len(population.species) if ENABLE_SPECIATION else 0
            current_mutation = INITIAL_MUTATION_RATE
            if DYNAMIC_MUTATION:
                progress = min(1.0, population.generation / MAX_GENERATIONS)
                current_mutation = INITIAL_MUTATION_RATE * (1 - progress * 0.6) + MIN_MUTATION_RATE
            
            best_agent = population.get_best_agent()
            texts = [
                f"Generation: {population.generation}/{MAX_GENERATIONS}",
                f"Best Fitness: {population.best_fitness:.1f}",
                f"Avg Fitness: {population.avg_fitness:.1f}",
                f"Alive: {sum(1 for a in population.agents if a.car.alive)}/{len(population.agents)}",
                f"Best Checkpoints: {best_agent.car.checkpoint_progress}",
                f"Unique Visited: {len(best_agent.car.checkpoint_visits)}",
                f"Best Forward: {best_agent.car.total_forward_distance:.1f}",
                f"Species: {species_count}" if ENABLE_SPECIATION else "",
                f"Mutation: {current_mutation:.3f}" if DYNAMIC_MUTATION else "",
                "",
                "Improvements:",
                f"+ Better checkpoint detection",
                f"+ Direction guidance sensor",
                f"+ Progressive reward system", 
                f"+ Adaptive mutation rates",
                "",
                "ESC to quit training"
            ]
        else:
            best_agent = population.get_best_agent()
            texts = [
                "Demo Mode - Best Trained Agent",
                f"Final Generation: {population.generation}",
                f"Best Fitness: {population.best_fitness:.1f}",
                f"Speed: {best_agent.car.speed:.1f}",
                f"Checkpoints: {best_agent.car.checkpoint_progress}",
                f"Unique Visited: {len(best_agent.car.checkpoint_visits)}",
                f"Distance: {best_agent.car.distance_traveled:.1f}",
                f"Forward: {best_agent.car.total_forward_distance:.1f}",
                "",
                "R to reset car",
                "ESC to quit"
            ]
        
        for text in texts:
            if text:
                surface = self.font.render(text, True, (255, 255, 255))
                self.screen.blit(surface, (10, y))
            y += 25
    
    def render(self, track: Track, population: Population, show_sensors: bool = False, training_mode: bool = True):
        """Enhanced rendering"""
        self.screen.fill((20, 40, 20))
        
        # Store track reference for progress indicators
        self.track = track
        
        # Draw track
        self.draw_track(track)
        
        # Draw cars
        best_agent = population.get_best_agent()
        
        if training_mode:
            # Show all cars during training
            for i, agent in enumerate(population.agents):
                if agent == best_agent:
                    self.draw_car(agent.car, (255, 255, 0))  # Best car in yellow
                    if show_sensors:
                        self.draw_sensors(agent.car)
                    self.draw_progress_indicators(agent.car)
                else:
                    alpha = 0.8 if agent.car.alive else 0.3
                    # Color based on progress
                    if agent.car.checkpoint_progress > 0:
                        color = (int(100 + 150 * alpha), int(255 * alpha), int(100 * alpha))
                    else:
                        color = (int(200 * alpha), int(200 * alpha), int(255 * alpha))
                    self.draw_car(agent.car, color)
        else:
            # Show only best car in demo mode
            self.draw_car(best_agent.car, (255, 255, 0))
            if show_sensors:
                self.draw_sensors(best_agent.car)
            self.draw_progress_indicators(best_agent.car)
        
        # Draw fitness graph
        if training_mode:
            self.draw_fitness_graph(population)
        
        # Draw UI
        self.draw_ui(population, training_mode)
        
        pygame.display.flip()
        return self.clock.tick(FPS)
    
    def check_events(self) -> dict:
        """Check for events and return action dict"""
        actions = {'quit': False, 'reset': False}
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                actions['quit'] = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    actions['quit'] = True
                elif event.key == pygame.K_r:
                    actions['reset'] = True
        
        return actions
