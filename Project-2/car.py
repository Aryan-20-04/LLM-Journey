from config import *
from track import Track
from vector2 import Vector2
from collections import deque
import math

class Car:
    def __init__(self, track: Track):
        self.track = track
        self.action_history = deque(maxlen=10)
        self.reverse_timer = 0.0
        self.reset()
        
    def reset(self):
        """Reset car to starting position"""
        self.pos = Vector2(self.track.start_pos.x, self.track.start_pos.y)
        self.velocity = Vector2(0, 0)
        self.angle = self.track.start_angle
        self.speed = 0.0
        self.alive = True
        
        # IMPROVED: Progress tracking
        self.distance_traveled = 0.0
        self.prev_pos = Vector2(self.pos.x, self.pos.y)
        self.last_checkpoint_index = -1
        self.checkpoint_progress = 0
        self.lap_time = 0.0
        self.collision_count = 0
        self.off_track_time = 0.0
        self.max_steer_angle = 90.0  # Reduced for smoother turning
        self.acceleration = 180.0
        
        # Track progress metrics
        self.total_forward_distance = 0.0
        self.backward_penalty = 0.0
        self.path_efficiency = 1.0
        self.last_partial_progress = 0.0   
        self.stagnation_time = 0.0
        
        # NEW: Direction-based progress tracking
        self.checkpoint_visits = set()  # Track which checkpoints we've hit
        self.wrong_direction_penalty = 0.0
        
        # Smooth driving metrics
        self.action_history.clear()
        self.steering_changes = 0
        self.throttle_changes = 0
        self.last_throttle = 0
        self.last_steering = 0
        
        # Initialize sensors
        self.sensor_readings = [1.0] * (N_SENSORS + 6)  # Added direction sensor
        self.update_sensors()

    def update(self, dt: float, throttle: float, steering: float):
        """IMPROVED: Better physics and progress tracking"""
        if not self.alive:
            return

        steering = max(-1, min(1, steering))
        throttle = max(-1, min(1, throttle))
        
        # Track action changes for smooth driving
        if abs(steering - self.last_steering) > 0.15:
            self.steering_changes += 1
        if abs(throttle - self.last_throttle) > 0.15:
            self.throttle_changes += 1
        self.last_steering = steering
        self.last_throttle = throttle

        # Update angle with smoother turning
        angle_change = steering * self.max_steer_angle * dt * math.pi / 180
        self.angle += angle_change

        # Calculate forward direction
        forward = Vector2(math.cos(self.angle), math.sin(self.angle))
        #if self.velocity.dot(forward) < 0:
         #   fitness -= 20 * abs(self.velocity.dot(forward))
        # Improved acceleration with better control
        if throttle > 0.1:
            # Forward acceleration
            accel_force = forward * throttle * self.acceleration * dt
            self.velocity = self.velocity + accel_force
        elif throttle < -0.1:
            # Braking/reverse
            if self.velocity.length() > 0:
                brake_force = self.velocity.normalize() * (-abs(throttle)) * self.acceleration * dt
                self.velocity = self.velocity + brake_force

        # Store previous position
        old_pos = Vector2(self.pos.x, self.pos.y)
        
        # Update position
        self.pos = self.pos + self.velocity * dt
        self.speed = self.velocity.length()

        # Calculate movement metrics
        movement_vector = self.pos - old_pos
        movement_distance = movement_vector.length()
        self.distance_traveled += movement_distance
        
        # Forward progress calculation
        if movement_distance > 0:
            forward_movement = movement_vector.dot(forward)
            if forward_movement > 0:
                self.total_forward_distance += forward_movement
            else:
                self.backward_penalty += abs(forward_movement) * 0.3

        # Apply friction
        self.velocity = self.velocity * FRICTION
        
        # Limit speed
        if self.speed > MAX_SPEED:
            self.velocity = self.velocity.normalize() * MAX_SPEED
            self.speed = MAX_SPEED
        
        # Update lap time
        self.lap_time += dt
        # Prevent long reverse driving
        forward_component = self.velocity.dot(forward)
        if forward_component < -0.5:  # going backwards relative to heading
            self.reverse_timer += dt
            if self.reverse_timer > 2.0:  # more than 2s in reverse
                self.alive = False
                self.crash_reason = "Drove backwards"
        else:
            self.reverse_timer = 0.0
        # Update sensors
        self.update_sensors()
        
        # Check collisions and track boundaries
        self.check_collisions(dt)
        
        # Update progress
        self.update_progress_improved()
    
    def update_sensors(self):
        """IMPROVED: Enhanced sensor system with direction guidance"""
        self.sensor_readings = []
        
        # Distance sensors with better distribution
        for i in range(N_SENSORS):
            angle_range = math.pi * 1.2  # Wider sensor range
            sensor_angle = self.angle + (i - N_SENSORS//2) * angle_range / (N_SENSORS - 1)
            direction = Vector2(math.cos(sensor_angle), math.sin(sensor_angle))
            
            distance = self.track.get_distance_to_wall(self.pos, direction, SENSOR_RANGE)
            normalized_distance = min(1.0, distance / SENSOR_RANGE)
            self.sensor_readings.append(normalized_distance)
        
        # Enhanced inputs
        self.sensor_readings.extend([
            min(1.0, self.speed / MAX_SPEED),  # Current speed
            math.sin(self.angle),              # Angle sine
            math.cos(self.angle),              # Angle cosine
            self.velocity.x / MAX_SPEED,       # Velocity X component
            self.velocity.y / MAX_SPEED,       # Velocity Y component
            self.get_checkpoint_direction()    # NEW: Direction to next checkpoint
        ])
    
    def get_checkpoint_direction(self) -> float:
        """NEW: Get normalized direction to next checkpoint"""
        if not self.track.checkpoints:
            return 0.0
            
        # Find next checkpoint
        next_checkpoint_idx = (self.last_checkpoint_index + 1) % len(self.track.checkpoints)
        next_checkpoint = self.track.checkpoints[next_checkpoint_idx]
        
        # Calculate direction to checkpoint
        to_checkpoint = next_checkpoint - self.pos
        if to_checkpoint.length() > 0:
            to_checkpoint = to_checkpoint.normalize()
            forward = Vector2(math.cos(self.angle), math.sin(self.angle))
            # Return dot product: 1 = same direction, -1 = opposite direction
            return to_checkpoint.dot(forward)
        
        return 0.0

    def check_collisions(self, dt: float):
        """IMPROVED: More forgiving collision detection"""
        if not self.track.is_on_track(self.pos):
            self.off_track_time += dt
            # More lenient off-track time
            if self.off_track_time > 4.0:
                self.alive = False
                self.crash_reason = "Off-track too long"
            
        else:
            # Faster recovery when back on track
            self.off_track_time = max(0, self.off_track_time - dt * 3)
        
        # More lenient wall collision detection
        min_sensor = min(self.sensor_readings[:N_SENSORS])
        if min_sensor < 0.02:  # Very close to wall
            self.collision_count += max(0,self.collision_count-dt*5)  # Decaying collisions over time
            # More forgiving collision threshold
            if self.collision_count > 30:
                self.alive = False
                self.crash_reason = "Wall collision (too many)"
                
        
    
    def update_progress_improved(self):
        """IMPROVED: Much better checkpoint progression system"""
        if not self.track.checkpoints:
            return
        
        # Check next expected checkpoint
        next_checkpoint_idx = (self.last_checkpoint_index + 1) % len(self.track.checkpoints)
        next_checkpoint_pos = self.track.checkpoints[next_checkpoint_idx]
        
        dist_to_next = (self.pos - next_checkpoint_pos).length()
        checkpoint_radius = CHECKPOINT_RADIUS
        
        new_partial = max(0.0, 1.0 - dist_to_next / checkpoint_radius)
        
        if not hasattr(self, 'last_partial_progress'):
            self.last_checkpoint_progress = 0.0
            self.stagnation_time = 0.0
        
        if new_partial > self.last_partial_progress + 0.01:  # Small improvement
            self.stagnation_time = 0.0
        else:
            self.stagnation_time += 1 / FPS  # approx per-frame time

        # Kill agent if stuck too long
        limit=6.0 + self.checkpoint_progress*1.5
        if self.stagnation_time > limit and self.speed < 1.0:  # 12 seconds no progress
            self.alive = False
            self.crash_reason = "Stagnated (no progress)"

        # Save new partial
        self.partial_progress = new_partial
        self.last_partial_progress = new_partial

        # Reached Checkpoint
        if dist_to_next < checkpoint_radius:
            expected_dir = None
            if next_checkpoint_idx < len(self.track.checkpoint_directions):
               expected_dir = self.track.checkpoint_directions[next_checkpoint_idx]
            current_dir = Vector2(math.cos(self.angle), math.sin(self.angle))

            if expected_dir is None or expected_dir.dot(current_dir) > -0.2:
                self.last_checkpoint_index = next_checkpoint_idx
                self.checkpoint_progress += 1
                self.checkpoint_visits.add(next_checkpoint_idx)
                
                if self.checkpoint_progress <= 3 or self.checkpoint_progress % 5 == 0:
                    print(f"✓ Checkpoint {next_checkpoint_idx} reached! Progress: {self.checkpoint_progress}")
                else:
                    self.wrong_direction_penalty += 0.5
                    
        # Bonus for visiting unique checkpoints
        unique_checkpoints_visited = len(self.checkpoint_visits)
        if unique_checkpoints_visited > self.checkpoint_progress // len(self.track.checkpoints):
            self.path_efficiency = min(1.5, 1.0 + unique_checkpoints_visited * 0.1)
        else:
            self.path_efficiency = 1.0
            
    def get_fitness(self) -> float:
        """IMPROVED: Better fitness function with multiple objectives"""
        if not self.alive and self.lap_time < 2.0:
            return 0.0  # Immediate death penalty
        
        fitness = 0.0
        
        # Base survival bonus
        fitness += min(self.lap_time * 40, 8000)
        
        # MAJOR: Distance-based progress (most important early on)
        fitness += min(self.total_forward_distance * 4.0, 4000)
        
        fitness += self.partial_progress * 2000

        # MAJOR: Checkpoint progress (exponentially more valuable)
        for i in range(min(self.checkpoint_progress, 20)):
            fitness += 1200 + (i * 300)  # Exponential reward
        

        # Unique checkpoint bonus
        fitness += len(self.checkpoint_visits) * 1500
        
        # Speed bonus when making progress
        if self.checkpoint_progress > 0:
            fitness += self.speed * self.checkpoint_progress * 20

        # Path efficiency bonus
        fitness *= self.path_efficiency
        
        # Stay At Center
        center=Vector2(SCREEN_W // 2, SCREEN_H // 2)
        dist_to_center = (self.pos - center).length()
        ideal_radius = 230
        center_deviation = abs(dist_to_center - ideal_radius)
        fitness -= center_deviation * 0.2

        # PENALTIES
        fitness -= self.backward_penalty * 5
        fitness -= self.collision_count * 20
        fitness -= self.off_track_time * 40
        fitness -= self.wrong_direction_penalty * 10
        
        # Penalize backward motion
        forward = Vector2(math.cos(self.angle), math.sin(self.angle))
        if self.velocity.dot(forward) < 0:
            fitness -= 40  # penalty for going backwards
        
        # Smooth driving bonus/penalty
        if SMOOTH_DRIVING_PENALTY and self.checkpoint_progress > 0:
            fitness -= self.steering_changes * 0.8
            fitness -= self.throttle_changes * 0.5
        
        # Huge bonus for completing laps
        if self.checkpoint_progress >= len(self.track.checkpoints):
            laps_completed = self.checkpoint_progress // len(self.track.checkpoints)
            fitness += laps_completed * 8000 # Massive lap completion bonus
            
            # Time bonus for faster laps
            if self.lap_time > 0:
                fitness += 2500 / max(10, self.lap_time)

        return max(0.0, min(fitness, 1_000_000.0))
