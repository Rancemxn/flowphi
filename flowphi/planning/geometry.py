import math
from shapely import Polygon
from flowphi.charts.model import Position, Vector

JUDGE_HALF_TAP = 0.106875
JUDGE_HALF_DRAG = 0.118125
JUDGE_MARGIN = 0.85


class JudgeArea:
    __slots__ = ('center', 'rotation', 'poly', 'half_width')

    def __init__(
        self, center: Position, rotation: Vector, screen_w: float, screen_h: float,
        half: float = JUDGE_HALF_TAP, scale: float = JUDGE_MARGIN
    ) -> None:
        self.center = center
        self.rotation = rotation
        self.half_width = screen_w * half * scale
        perp = rotation * 1j
        limit = math.hypot(screen_w, screen_h) + abs(center)
        d_rot = rotation * self.half_width
        d_perp = perp * limit
        c1 = center + d_rot + d_perp
        c2 = center + d_rot - d_perp
        c3 = center - d_rot - d_perp
        c4 = center - d_rot + d_perp
        self.poly = Polygon([
            (c1.real, c1.imag),
            (c2.real, c2.imag),
            (c3.real, c3.imag),
            (c4.real, c4.imag)
        ])

    def get_valid_poly(self, screen_poly: Polygon, pause_poly: Polygon) -> Polygon:
        inter = self.poly.intersection(screen_poly)
        if inter.is_empty:
            return Polygon()
        valid = inter.difference(pause_poly)
        return valid if not valid.is_empty else Polygon()

    def is_valid_zone(self, valid_poly: Polygon, screen_w: float) -> bool:
        if valid_poly.is_empty:
            return False
        return valid_poly.area > (screen_w * 1e-4) ** 2
