# coating_simulator_project/coating_simulator/visualization/geom_source.py
"""
Functions for plotting the 2D projections of the source geometry.
Функции для отрисовки 2D-проекций геометрии источника.
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Используем относительные импорты
from ..core.distribution import rotation_matrix
from .. import config

# --- Helper Function ---
def _apply_rotation(points_local: np.ndarray, orient_matrix: np.ndarray, base_pos: np.ndarray) -> np.ndarray:
    """
    Applies orientation and translation to local points.
    Ensures result is (3, N) for multiple points or (3,) for a single point.
    Применяет ориентацию и смещение к локальным точкам.
    Гарантирует результат (3, N) для нескольких точек или (3,) для одной точки.
    """
    points_local = np.asarray(points_local, dtype=float)
    orient_matrix = np.asarray(orient_matrix, dtype=float)
    base_pos = np.asarray(base_pos, dtype=float)

    # Ensure points_local is 2D (3, N)
    if points_local.ndim == 1:
        if points_local.shape == (3,):
            points_local = points_local[:, np.newaxis] # Convert (3,) to (3, 1)
        else:
            raise ValueError(f"Invalid shape for 1D local points: {points_local.shape}. Expected (3,).")
    elif points_local.shape[0] != 3:
        # Try transposing if shape is (N, 3)
        if points_local.shape[1] == 3:
            points_local = points_local.T
        else:
             raise ValueError(f"Invalid shape for 2D local points: {points_local.shape}. Expected (3, N).")

    # Ensure base_pos is 1D (3,) before reshaping for addition
    if base_pos.shape != (3,):
         if base_pos.size == 3:
             base_pos = base_pos.flatten()
         else:
              raise ValueError(f"Invalid shape for base_pos: {base_pos.shape}. Expected (3,).")
    
    # Perform rotation and translation
    # orient_matrix (3, 3) @ points_local (3, N) -> result (3, N)
    # Add base_pos (3,) which broadcasts to (3, N)
    points_global = orient_matrix @ points_local + base_pos[:, np.newaxis] # Add base_pos as column vector

    # If the original input was effectively a single point, return a 1D array (3,)
    if points_global.shape[1] == 1:
        return points_global.flatten()
    else:
        return points_global # Return as (3, N)


# --- Plotting Function ---
def plot_source_2d(ax_top: plt.Axes, ax_side: plt.Axes, params: dict):
    """Plots the source's 2D projections."""
    src_type = params.get('src_type')
    src_x = params.get('src_x', 0.0)
    src_y = params.get('src_y', 0.0)
    src_z = params.get('src_z', 100.0) 
    rot_x_deg = params.get('rot_x', 0.0)
    rot_y_deg = params.get('rot_y', 0.0)
    color = 'red'
    alpha = 0.6 
    linewidth = 1.5 

    rot_x_rad = math.radians(rot_x_deg)
    rot_y_rad = math.radians(rot_y_deg)
    # Consistent rotation order: Apply Y then X (matches geom_emission)
    mat_rot_y = rotation_matrix(np.array([0, 1, 0]), rot_y_rad)
    mat_rot_x = rotation_matrix(np.array([1, 0, 0]), rot_x_rad)
    source_orient_matrix = mat_rot_x @ mat_rot_y 
    base_pos = np.array([src_x, src_y, src_z]) # Should be (3,)

    # --- Top View (XY) ---
    ax_top.plot(src_x, src_y, 'o', color=color, markersize=5, label='Источник (центр)')

    if src_type == config.SOURCE_RING or src_type == config.SOURCE_CIRCULAR:
        src_diameter = params.get('src_diameter', 10.0)
        src_radius = src_diameter / 2.0
        theta = np.linspace(0, 2 * np.pi, 50) 
        x_local = src_radius * np.cos(theta)
        y_local = src_radius * np.sin(theta)
        z_local = np.zeros_like(theta) 
        
        # points_local is (3, 50)
        points_local = np.vstack((x_local, y_local, z_local))
        # points_global should be (3, 50)
        points_global = _apply_rotation(points_local, source_orient_matrix, base_pos)
        
        label_src_shape = f'Источник ({src_type})'
        if src_type == config.SOURCE_CIRCULAR:
             ax_top.fill(points_global[0,:], points_global[1,:], color=color, alpha=alpha*0.5, label=label_src_shape)
             ax_top.plot(points_global[0,:], points_global[1,:], '-', color=color, linewidth=linewidth) 
        else: # Ring
             ax_top.plot(points_global[0,:], points_global[1,:], '-', color=color, linewidth=linewidth, label=label_src_shape)

    elif src_type == config.SOURCE_LINEAR:
        src_length = params.get('src_length', 20.0)
        src_angle_deg = params.get('src_angle', 0.0) 
        
        # Local endpoints along source's X-axis
        x_local_line_initial = np.array([-src_length/2, src_length/2])
        y_local_line_initial = np.zeros(2)
        z_local_line_initial = np.zeros(2)
        points_local_initial = np.vstack((x_local_line_initial, y_local_line_initial, z_local_line_initial)) # Shape (3, 2)
        
        # Apply local rotation src_angle around local Z-axis
        rot_src_angle_rad = math.radians(src_angle_deg)
        mat_rot_src_angle = rotation_matrix(np.array([0,0,1]), rot_src_angle_rad)
        points_local_oriented = mat_rot_src_angle @ points_local_initial # Shape (3, 2)
        
        # Apply main orientation and translation
        # points_global_line should be (3, 2)
        points_global_line = _apply_rotation(points_local_oriented, source_orient_matrix, base_pos)
        ax_top.plot(points_global_line[0,:], points_global_line[1,:], '-', color=color, linewidth=linewidth + 0.5, label='Источник (Линия)') 

    # --- Side View (XZ) ---
    ax_side.plot(src_x, src_z, 'o', color=color, markersize=5, label='Источник (центр)') 

    if src_type == config.SOURCE_RING or src_type == config.SOURCE_CIRCULAR:
        src_diameter = params.get('src_diameter', 10.0)
        src_radius = src_diameter / 2.0
        theta = np.linspace(0, 2 * np.pi, 50)
        x_local = src_radius * np.cos(theta)
        y_local = src_radius * np.sin(theta) 
        z_local = np.zeros_like(theta)
        
        points_local = np.vstack((x_local, y_local, z_local)) # Shape (3, 50)
        points_global = _apply_rotation(points_local, source_orient_matrix, base_pos) # Shape (3, 50)
        
        label_src_shape_side = f'Источник ({src_type})' 
        # Plot XZ projection
        ax_side.plot(points_global[0,:], points_global[2,:], '-', color=color, linewidth=linewidth, label=label_src_shape_side)

    elif src_type == config.SOURCE_LINEAR:
        src_length = params.get('src_length', 20.0)
        src_angle_deg = params.get('src_angle', 0.0)
        
        x_local_line_initial = np.array([-src_length/2, src_length/2])
        y_local_line_initial = np.zeros(2)
        z_local_line_initial = np.zeros(2)
        points_local_initial = np.vstack((x_local_line_initial, y_local_line_initial, z_local_line_initial)) # Shape (3, 2)
        
        rot_src_angle_rad = math.radians(src_angle_deg)
        mat_rot_src_angle = rotation_matrix(np.array([0,0,1]), rot_src_angle_rad)
        points_local_oriented = mat_rot_src_angle @ points_local_initial # Shape (3, 2)
        
        points_global_line = _apply_rotation(points_local_oriented, source_orient_matrix, base_pos) # Shape (3, 2)
        # Plot XZ projection
        ax_side.plot(points_global_line[0,:], points_global_line[2,:], '-', color=color, linewidth=linewidth + 0.5, label='Источник (Линия)')

    # --- Plot source orientation vector projection (arrow) ---
    arrow_visualization_length = src_z / 2.0 if src_z > 10 else 5.0 # Adjust length based on height
    if arrow_visualization_length <= 0: arrow_visualization_length = 20.0 

    local_dir_vec_for_arrow = np.array([0, 0, -1]) * arrow_visualization_length # Shape (3,)
    
    # global_dir_vec_end should be (3,)
    global_dir_vec_end = _apply_rotation(local_dir_vec_for_arrow, source_orient_matrix, base_pos) 
    
    arrow_label = 'Направление источника'
    head_w = max(1.0, arrow_visualization_length * 0.1)
    head_l = max(1.5, arrow_visualization_length * 0.15)
    
    # Ensure start and end points are 1D for arrow calculation
    start_arrow_top = base_pos[:2] # (x, y)
    end_arrow_top = global_dir_vec_end[:2] # (x, y)
    start_arrow_side = np.array([base_pos[0], base_pos[2]]) # (x, z)
    end_arrow_side = np.array([global_dir_vec_end[0], global_dir_vec_end[2]]) # (x, z)

    # Check if arrow has length before drawing
    if np.linalg.norm(end_arrow_top - start_arrow_top) > 1e-3:
        ax_top.arrow(start_arrow_top[0], start_arrow_top[1], 
                     end_arrow_top[0]-start_arrow_top[0], end_arrow_top[1]-start_arrow_top[1],
                     head_width=head_w, head_length=head_l, 
                     fc='darkred', ec='darkred', length_includes_head=True, label=arrow_label)
    
    if np.linalg.norm(end_arrow_side - start_arrow_side) > 1e-3:
        ax_side.arrow(start_arrow_side[0], start_arrow_side[1], 
                      end_arrow_side[0]-start_arrow_side[0], end_arrow_side[1]-start_arrow_side[1],
                      head_width=head_w, head_length=head_l, 
                      fc='darkred', ec='darkred', length_includes_head=True, label=arrow_label)

