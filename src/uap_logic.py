import math
from typing import List, Tuple

class UAPAnalyzer:
    def __init__(self):
        self.max_turn_angle = 45.0

    def calculate_angle(self, p1, p2, p3) -> float:
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        if mag1 * mag2 == 0: return 0.0
        cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_angle))

    def check_sharp_turns(self, trajectory: List[Tuple[int, int]]) -> bool:
        if len(trajectory) < 3: return False
        for i in range(len(trajectory) - 2):
            angle = self.calculate_angle(trajectory[i], trajectory[i+1], trajectory[i+2])
            if angle < (180 - self.max_turn_angle): return True
        return False
