from config import *
from vector2 import Vector2
import random,math
from typing import List,Optional

class Track:
    def __init__(self, track_type="oval", seed: Optional[int]=None):
        self.waypoints = []
        self.walls = []
        self.checkpoints : List[Vector2]= []
        self.checkpoint_directions :List[Vector2] = []  # NEW: Direction vectors for checkpoints
        self.start_pos = Vector2(0, 0)
        self.start_angle = 0
        self.track_type = track_type
        
        self.seed= seed
        if seed is not None:
            random.seed(seed)
            
        if track_type == "oval":
            self.generate_oval_track()
        elif track_type == "figure8":
            self.generate_figure8_track()
        elif track_type == "random":
            self.generate_random_track()
    
    def generate_oval_track(self):
        """Generate a smooth oval track with better checkpoint placement"""
        center_x, center_y = SCREEN_W // 2, SCREEN_H // 2
        
        # Create oval waypoints
        num_points = 32
        radius_x = 280
        radius_y = 180
        
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            x = center_x + radius_x * math.cos(angle)
            y = center_y + radius_y * math.sin(angle)
            self.waypoints.append(Vector2(x, y))
        
        # Create walls from waypoints
        self.create_walls_from_waypoints()
        
        # Set start position - moved further from first checkpoint
        self.start_pos = Vector2(center_x + radius_x - 80, center_y)
        self.start_angle = -math.pi / 2
        
        # Create better spaced checkpoints
        self.create_checkpoints_improved()
    
    def generate_figure8_track(self):
        """Generate a figure-8 track for more complex navigation"""
        center_x, center_y = SCREEN_W // 2, SCREEN_H // 2
        num_points = 48
        
        for i in range(num_points):
            t = (i / num_points) * 4 * math.pi
            # Figure-8 parametric equations
            x = center_x + 150 * math.sin(t)
            y = center_y + 100 * math.sin(t) * math.cos(t)
            self.waypoints.append(Vector2(x, y))
        
        self.create_walls_from_waypoints()
        self.start_pos = Vector2(center_x + 150, center_y)
        self.start_angle = math.pi / 2
        self.create_checkpoints_improved()
    
    def generate_random_track(self):
        """Generate a random track layout"""
        center_x, center_y = SCREEN_W // 2, SCREEN_H // 2
        num_points = 24
        base_radius = 200
        
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            # Add random variation to radius
            radius_variation = random.uniform(0.7, 1.3)
            radius = base_radius * radius_variation
            
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle) * 0.8  # Slightly oval
            self.waypoints.append(Vector2(x, y))
        
        self.create_walls_from_waypoints()
        self.start_pos = Vector2(center_x + base_radius - 50, center_y)
        self.start_angle = -math.pi / 2
        self.create_checkpoints_improved()
    
    def create_walls_from_waypoints(self):
        """Create inner and outer walls from waypoints"""
        self.walls = []
        
        for i in range(len(self.waypoints)):
            p1 = self.waypoints[i]
            p2 = self.waypoints[(i + 1) % len(self.waypoints)]
            
            # Calculate perpendicular direction
            direction = (p2 - p1).normalize()
            perp = Vector2(-direction.y, direction.x)
            
            # Inner wall
            inner1 = p1 + perp * (TRACK_WIDTH // 2)
            inner2 = p2 + perp * (TRACK_WIDTH // 2)
            self.walls.append((inner1, inner2))
            
            # Outer wall
            outer1 = p1 - perp * (TRACK_WIDTH // 2)
            outer2 = p2 - perp * (TRACK_WIDTH // 2)
            self.walls.append((outer1, outer2))
    
    def create_checkpoints_improved(self):
        """IMPROVED: Create checkpoints with proper spacing and direction tracking"""
        self.checkpoints = []
        self.checkpoint_directions = []
        
        # Create fewer, better spaced checkpoints
        num_checkpoints = NUM_CHECKPOINTS # Reduced from 8
        checkpoint_interval = len(self.waypoints) // num_checkpoints
        
        for i in range(num_checkpoints):
            waypoint_idx = (i * checkpoint_interval) % len(self.waypoints)
            checkpoint_pos = self.waypoints[waypoint_idx]
            self.checkpoints.append(checkpoint_pos)
            
            # Calculate direction to next checkpoint
            next_waypoint_idx = ((i + 1) * checkpoint_interval) % len(self.waypoints)
            next_checkpoint_pos = self.waypoints[next_waypoint_idx]
            direction = (next_checkpoint_pos - checkpoint_pos).normalize()
            self.checkpoint_directions.append(direction)
        
        print(f"Created {len(self.checkpoints)} checkpoints for {self.track_type} track")
    
    def get_distance_to_wall(self, start: Vector2, direction: Vector2, max_distance: float) -> float:
        """Cast a ray and return distance to nearest wall"""
        end = start + direction * max_distance
        min_distance = max_distance
        
        for wall_start, wall_end in self.walls:
            intersection = self.line_intersection(start, end, wall_start, wall_end)
            if intersection:
                distance = (intersection - start).length()
                min_distance = min(min_distance, distance)
        
        return min_distance
    
    def line_intersection(self, p1: Vector2, p2: Vector2, p3: Vector2, p4: Vector2) -> Optional[Vector2]:
        """Find intersection point between two line segments"""
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
        x3, y3 = p3.x, p3.y
        x4, y4 = p4.x, p4.y
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return Vector2(x, y)
        
        return None
    
    def is_on_track(self, pos: Vector2) -> bool:
        """Check if position is on the track"""
        center = Vector2(SCREEN_W // 2, SCREEN_H // 2)
        dist_to_center = (pos - center).length()

        if self.track_type == "oval":
            return 140 < dist_to_center < 320
        elif self.track_type == "figure8":
            return dist_to_center < 280
        else:  # random
            return 80 < dist_to_center < 350
