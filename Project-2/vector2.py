import math
from dataclasses import dataclass

@dataclass
class Vector2:
    x: float
    y: float
    
    def __add__(self, other:'Vector2')->'Vector2':
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other:'Vector2')->'Vector2':
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> 'Vector2':
        return Vector2(self.x * scalar, self.y * scalar)

    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2)

    def normalize(self) -> 'Vector2':
        length = self.length()
        if length > 0:
            return Vector2(self.x / length, self.y / length)
        return Vector2(0, 0)
    
    def rotate(self, angle):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Vector2(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a
        )
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y
