# coating_simulator_project/coating_simulator/visualization/geom_utils.py
"""
Utility functions for geometry calculations used in visualization modules.
Вспомогательные функции для геометрических вычислений, используемые в модулях визуализации.
"""
import numpy as np
import math
# --- ДОБАВЛЕН ИМПОРТ config ---
from .. import config 
# -----------------------------

def _cross_product_z(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """ 2D cross product (magnitude of Z component). o, a, b are 1D np.arrays or tuples. """
    o = np.asarray(o); a = np.asarray(a); b = np.asarray(b)
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def _convex_hull(points_tuples: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    Computes the convex hull of a set of 2D points using Andrew's monotone chain algorithm.
    Args:
        points_tuples: A list of (x, y) tuples.
    Returns:
        A list of (x, y) tuples representing the convex hull vertices in counter-clockwise order.
        Returns an empty list if input has less than 3 unique points after filtering.
    """
    unique_points_tuples = list(set(points_tuples))
    if not unique_points_tuples or len(unique_points_tuples) < 3:
        return [tuple(p) for p in np.array(unique_points_tuples)] if unique_points_tuples else []

    points = np.array(unique_points_tuples)
    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    
    upper_hull_np = []
    for p_np in points:
        while len(upper_hull_np) >= 2 and _cross_product_z(upper_hull_np[-2], upper_hull_np[-1], p_np) <= 0:
            upper_hull_np.pop()
        upper_hull_np.append(p_np)

    lower_hull_np = []
    for p_np in reversed(points): 
        while len(lower_hull_np) >= 2 and _cross_product_z(lower_hull_np[-2], lower_hull_np[-1], p_np) <= 0:
            lower_hull_np.pop()
        lower_hull_np.append(p_np)

    combined_hull_np_arrays = upper_hull_np[:-1] + lower_hull_np[:-1]
    
    if len(combined_hull_np_arrays) < 3:
         return [tuple(p) for p in points] 

    return [tuple(p_np) for p_np in combined_hull_np_arrays]


def _generate_circle_boundary(center_xy: np.ndarray, radius: float, num_points: int = 32) -> list[tuple[float, float]]:
    """ Generates points on the boundary of a circle. """
    center_xy = np.asarray(center_xy) 
    if radius <= 1e-6: 
        return [(float(center_xy[0]), float(center_xy[1]))]
    points = []
    angles = np.linspace(0, 2 * math.pi, num_points, endpoint=False)
    for angle in angles:
        points.append((float(center_xy[0] + radius * math.cos(angle)), 
                       float(center_xy[1] + radius * math.sin(angle))))
    return points

def _is_point_in_polygon(point_xy: tuple[float, float], polygon_vertices_xy: list[tuple[float, float]]) -> bool:
    """
    Checks if a point is inside a polygon using the Ray Casting algorithm.
    """
    x, y = point_xy
    n = len(polygon_vertices_xy)
    if n < 3:
        return False 

    inside = False
    p1x, p1y = polygon_vertices_xy[0]
    for i in range(n + 1):
        p2x, p2y = polygon_vertices_xy[i % n]
        if y > min(p1y, p2y):            
            if y <= max(p1y, p2y):       
                if x <= max(p1x, p2x):   
                    if p1y != p2y:       
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside 
        p1x, p1y = p2x, p2y 
    return inside

def _generate_substrate_boundary(tgt_type: str, params: dict, num_sub_points: int = 128) -> list[tuple[float, float]]:
    """ Generates boundary points for the substrate. Ensures result is list of tuples. """
    pts_tuples = []
    # Используем константы из импортированного config
    if tgt_type in (config.TARGET_DISK, config.TARGET_DOME, config.TARGET_PLANETARY):
        R_substrate = params.get('diameter', 100.0) / 2.0
        pts_tuples = _generate_circle_boundary(np.array([0.0, 0.0]), R_substrate, num_sub_points)
    elif tgt_type == config.TARGET_LINEAR:
        L_substrate = params.get('length', 100.0) 
        W_substrate = params.get('width', 50.0)   
        half_L_sub, half_W_sub = L_substrate / 2.0, W_substrate / 2.0
        pts_tuples.extend([
            (float(-half_L_sub), float(-half_W_sub)), (float(half_L_sub), float(-half_W_sub)),
            (float(half_L_sub), float(half_W_sub)), (float(-half_L_sub), float(half_W_sub))
        ])
    return pts_tuples 

# --- НОВАЯ ФУНКЦИЯ ---
def _polygon_area(vertices: list[tuple[float, float]]) -> float:
    """ Calculates the area of a polygon given its vertices using Shoelace formula. """
    if not vertices or len(vertices) < 3:
        return 0.0
    x = np.array([v[0] for v in vertices])
    y = np.array([v[1] for v in vertices])
    # Apply Shoelace formula: 0.5 * |(x1*y2 + x2*y3 + ... + xn*y1) - (y1*x2 + y2*x3 + ... + yn*x1)|
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
# --------------------
