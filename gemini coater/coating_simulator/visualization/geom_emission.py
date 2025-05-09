# coating_simulator_project/coating_simulator/visualization/geom_emission.py
"""
Main module for plotting 2D emission geometry. Orchestrates calls to specific
source type emission plotting functions. Includes helper functions.
Основной модуль для отрисовки 2D-геометрии эмиссии. Оркеструет вызовы
специализированных функций отрисовки для каждого типа источника. Включает вспомогательные функции.
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import traceback 

# Используем относительные импорты
from ..core.distribution import rotation_matrix 
from .. import config
# --- ИМПОРТ ГЕОМЕТРИЧЕСКИХ УТИЛИТ ИЗ НОВОГО ФАЙЛА ---
# (Убедитесь, что geom_utils.py находится в той же папке 'visualization')
try:
    from .geom_utils import _convex_hull # Import only what's needed if used here
except ImportError as e_util:
     print(f"ERROR: Failed to import from geom_utils.py: {e_util}")
     # Define dummy if needed, though _convex_hull might not be directly used in *this* file anymore
     def _convex_hull(*args, **kwargs): print("Error: _convex_hull not loaded"); return []

# --- Вспомогательные функции ---
def _intersect_z0(start_point: np.ndarray, direction: np.ndarray) -> np.ndarray | None:
    """
    Calculates intersection point of a ray with Z=0 plane.
    Handles potential multi-dimensional array errors by validating input shapes.
    Рассчитывает точку пересечения луча с плоскостью Z=0.
    Обрабатывает потенциальные ошибки многомерных массивов путем проверки входных форм.
    """
    try:
        start_point = np.asarray(start_point, dtype=float)
        direction = np.asarray(direction, dtype=float)

        if start_point.shape != (3,):
            print(f"ERROR (_intersect_z0): Invalid start_point shape. Expected (3,), got {start_point.shape}. Value: {start_point}")
            return None
        if direction.shape != (3,):
            if direction.size == 3:
                 print(f"Warning (_intersect_z0): Flattening direction shape from {direction.shape} to (3,).")
                 direction = direction.flatten()
            else:
                print(f"ERROR (_intersect_z0): Invalid direction shape. Expected (3,), got {direction.shape}. Value: {direction}")
                return None
            if direction.shape != (3,): # Check again after potential flatten
                 print(f"ERROR (_intersect_z0): Direction shape still invalid after flatten: {direction.shape}")
                 return None

        dir_z = direction[2] 

        if abs(dir_z) < 1e-9: 
            return None 

        t_intersect = -start_point[2] / dir_z 

        if t_intersect >= -1e-6 : 
            intersection_point = start_point + direction * t_intersect
            if intersection_point.shape == (3,):
                 return intersection_point
            else:
                 print(f"ERROR (_intersect_z0): Calculated intersection point has unexpected shape: {intersection_point.shape}")
                 return None
        else:
            return None

    except Exception as e:
        print(f"ERROR (_intersect_z0): Unexpected error during calculation: {e}")
        print(f"  start_point: {start_point}, shape: {getattr(start_point, 'shape', 'N/A')}")
        print(f"  direction: {direction}, shape: {getattr(direction, 'shape', 'N/A')}")
        traceback.print_exc()
        return None


def get_rotation_matrix_align_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """
    Computes the rotation matrix that aligns vector v_from with vector v_to.
    Handles edge cases like zero vectors, parallel, and anti-parallel vectors.
    Вычисляет матрицу поворота, которая совмещает вектор v_from с вектором v_to.
    Обрабатывает граничные случаи, такие как нулевые векторы, параллельные и антипараллельные векторы.
    """
    v_from = np.asarray(v_from, dtype=float)
    v_to = np.asarray(v_to, dtype=float)

    norm_from = np.linalg.norm(v_from)
    norm_to = np.linalg.norm(v_to)

    if norm_from < 1e-9 or norm_to < 1e-9:
        return np.identity(3) 

    v_from_unit = v_from / norm_from
    v_to_unit = v_to / norm_to

    dot_product = np.dot(v_from_unit, v_to_unit)

    if np.isclose(dot_product, 1.0):
        return np.identity(3)

    if np.isclose(dot_product, -1.0):
        ortho_axis = np.cross(v_from_unit, [1.0, 0.0, 0.0])
        if np.linalg.norm(ortho_axis) < 1e-9: 
            ortho_axis = np.cross(v_from_unit, [0.0, 1.0, 0.0])
        
        if np.linalg.norm(ortho_axis) > 1e-9:
             ortho_axis = ortho_axis / np.linalg.norm(ortho_axis)
        else: 
             ortho_axis = np.array([1.0, 0.0, 0.0]) 
             
        return rotation_matrix(ortho_axis, math.pi)

    axis = np.cross(v_from_unit, v_to_unit)
    angle = math.acos(np.clip(dot_product, -1.0, 1.0)) 
    
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        return np.identity(3) if dot_product > 0 else rotation_matrix([1,0,0], math.pi) 

    return rotation_matrix(axis / axis_norm, angle)


# --- Импорт специализированных функций отрисовки ---
# These should now import without causing circular dependencies
try:
    from .geom_emission_ring import plot_ring_emission
    from .geom_emission_circular import plot_circular_emission
    from .geom_emission_point import plot_point_emission
    from .geom_emission_linear import plot_linear_emission
    print("Successfully imported specific emission plotters.") # Debug print
except ImportError as e:
     print(f"ERROR: Failed to import specific emission plotters in geom_emission.py: {e}")
     # Define dummy functions to prevent crashes later if imports fail
     def plot_ring_emission(*args, **kwargs): print("Error: plot_ring_emission not loaded")
     def plot_circular_emission(*args, **kwargs): print("Error: plot_circular_emission not loaded")
     def plot_point_emission(*args, **kwargs): print("Error: plot_point_emission not loaded")
     def plot_linear_emission(*args, **kwargs): print("Error: plot_linear_emission not loaded")


# --- Основная функция отрисовки эмиссии ---
def plot_emission_2d(ax_top: plt.Axes, ax_side: plt.Axes, params: dict):
    """
    Plots the 2D projections of the emission based on source type.
    Отображает 2D-проекции эмиссии в зависимости от типа источника.
    """
    try:
        src_type = params.get('src_type')
        src_x = params.get('src_x', 0.0)
        src_y = params.get('src_y', 0.0)
        src_z = params.get('src_z', 100.0)
        rot_x_deg = params.get('rot_x', 0.0)
        rot_y_deg = params.get('rot_y', 0.0)

        max_theta_half_deg = params.get('max_theta', 30.0) 
        max_theta_rad = math.radians(max_theta_half_deg)
        max_theta_rad = np.clip(max_theta_rad, 0, math.pi / 2.0) 
        full_angle_deg_for_label = math.degrees(max_theta_rad * 2.0)

        _rot_x_rad = math.radians(rot_x_deg)
        _rot_y_rad = math.radians(rot_y_deg)
        _mat_rot_x = rotation_matrix([1,0,0], _rot_x_rad)
        _mat_rot_y = rotation_matrix([0,1,0], _rot_y_rad)
        _source_orient_matrix = _mat_rot_x @ _mat_rot_y # Apply Y then X

        _emission_axis_global = _source_orient_matrix @ np.array([0,0,-1])

        common_data = {
            'source_center_global': np.array([src_x, src_y, src_z]),
            'source_orient_matrix': _source_orient_matrix,
            'emission_axis_global': _emission_axis_global,
            'max_theta_rad': max_theta_rad, 
            'full_angle_deg_for_label': full_angle_deg_for_label,
            'colors': {
                'emission': 'orange', 
                'emission_fill': '#FFDAB9', 
                'focus': 'purple', 
                'axis': '#FF4500', 
                'source_main': 'darkred', 
            },
            'alphas': {
                'general_line': 0.75,         
                'ring_edge_spread': 0.6,      
                'circular_edge_spread': 0.65, 
                'axis_center': 0.9,          
                'emission_area_fill': 0.4,   
            },
            'linewidths': {
                'ring_edge_spread': 1.0,
                'circular_edge_spread': 1.0,
                'polygon_outline': 1.2, 
                'axis_center': 1.2,
                'spread_center': 1.5, 
                'source_main': 2.0,   
            }
        }

        plotter_func = None
        if src_type == config.SOURCE_RING:
            plotter_func = plot_ring_emission
        elif src_type == config.SOURCE_CIRCULAR:
            plotter_func = plot_circular_emission
        elif src_type == config.SOURCE_POINT:
            plotter_func = plot_point_emission
        elif src_type == config.SOURCE_LINEAR:
            plotter_func = plot_linear_emission
        else:
            print(f"Предупреждение: Неизвестный тип источника '{src_type}' для отрисовки эмиссии.")

        if plotter_func:
            plotter_func(ax_top, ax_side, params, common_data)
            
    except Exception as e:
         print(f"ERROR in plot_emission_2d: {e}")
         traceback.print_exc()
