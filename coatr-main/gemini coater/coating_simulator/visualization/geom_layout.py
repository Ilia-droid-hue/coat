# coating_simulator_project/coating_simulator/visualization/geom_layout.py
"""
Functions for setting up the 2D geometry plot layout and orchestrating the drawing.
Includes precise auto-scaling based on plotted elements.
Функции для настройки макета 2D-графика геометрии и оркестровки отрисовки.
Включает точное автомасштабирование на основе отображаемых элементов.
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import OrderedDict

# Используем относительные импорты
from .geom_target import plot_target_2d
from .geom_source import plot_source_2d, _apply_rotation
from .geom_emission import plot_emission_2d # Removed _intersect_z0 as it's internal to geom_emission
from ..core.distribution import rotation_matrix
from .. import config

def _get_object_bounds(params: dict) -> tuple:
    """
    Calculates the precise min/max coordinates required to encompass
    the target, source, focus point, and emission intersection at Z=0.
    Рассчитывает точные мин/макс координаты, необходимые для охвата
    мишени, источника, точки фокуса и пересечения эмиссии с Z=0.

    Returns:
        tuple: (min_x, max_x, min_y, max_y, min_z, max_z)
    """
    target_type = params.get('target_type')
    src_type = params.get('src_type')
    src_x = params.get('src_x', 0.0)
    src_y = params.get('src_y', 0.0)
    src_z = params.get('src_z', 100.0)
    rot_x_deg = params.get('rot_x', 0.0)
    rot_y_deg = params.get('rot_y', 0.0)

    all_x = [0.0, src_x] # Start with origin and source center X
    all_y = [0.0, src_y] # Start with origin and source center Y
    all_z = [0.0, src_z] # Start with target plane (Z=0, peak of dome) and source height

    # --- Target Bounds ---
    if target_type == config.TARGET_DISK:
        radius = params.get('diameter', 100.0) / 2.0
        all_x.extend([-radius, radius]); all_y.extend([-radius, radius])
    elif target_type == config.TARGET_DOME:
        diameter = params.get('diameter', 100.0); dome_radius = params.get('dome_radius', 50.0)
        base_radius = diameter / 2.0; x_limit = min(base_radius, dome_radius)
        all_x.extend([-x_limit, x_limit]); all_y.extend([-x_limit, x_limit])
        min_z_dome = 0
        if dome_radius > 0:
            # Formula: z = dome_radius - sqrt(dome_radius^2 - x^2)
            # Lowest point is when x = x_limit
            sqrt_arg_limit = dome_radius**2 - x_limit**2
            min_z_dome = dome_radius - math.sqrt(sqrt_arg_limit) if sqrt_arg_limit >= 0 else 0
        all_z.append(min_z_dome)
    elif target_type == config.TARGET_LINEAR:
        length = params.get('length', 200.0); width = params.get('width', 100.0)
        all_x.extend([-length/2, length/2]); all_y.extend([-width/2, width/2])
    elif target_type == config.TARGET_PLANETARY:
        orbit_radius = params.get('orbit_diameter', 200.0) / 2.0
        planet_radius = params.get('planet_diameter', 50.0) / 2.0
        extent = orbit_radius + planet_radius
        all_x.extend([-extent, extent]); all_y.extend([-extent, extent])

    # --- Source Bounds ---
    rot_x_rad = math.radians(rot_x_deg); rot_y_rad = math.radians(rot_y_deg)
    mat_rot_y = rotation_matrix([0, 1, 0], rot_y_rad); mat_rot_x = rotation_matrix([1, 0, 0], rot_x_rad)
    source_orient_matrix = mat_rot_x @ mat_rot_y
    base_pos = np.array([src_x, src_y, src_z])

    if src_type == config.SOURCE_RING or src_type == config.SOURCE_CIRCULAR:
        src_radius = params.get('src_diameter', 10.0) / 2.0
        r_calc = src_radius if src_radius > 0 else 0.1 # Avoid zero for calculation
        # Approximate bounds by transforming corners of a bounding box around the source disk/ring
        corners_local = np.array([
            [r_calc, r_calc, -r_calc, -r_calc, 0], # Include center point
            [r_calc, -r_calc, r_calc, -r_calc, 0],
            [0, 0, 0, 0, 0]
        ])
        corners_global = _apply_rotation(corners_local, source_orient_matrix, base_pos)
        all_x.extend(corners_global[0,:].tolist()); all_y.extend(corners_global[1,:].tolist()); all_z.extend(corners_global[2,:].tolist())
    elif src_type == config.SOURCE_LINEAR:
        src_length = params.get('src_length', 20.0)
        src_angle_deg = params.get('src_angle', 0.0)
        rot_src_angle_rad = math.radians(src_angle_deg)
        mat_rot_src_angle = rotation_matrix([0,0,1], rot_src_angle_rad)
        ends_local = [np.array([-src_length/2, 0, 0]), np.array([0,0,0]), np.array([src_length/2, 0, 0])] # Include center
        ends_local_oriented = [mat_rot_src_angle @ p for p in ends_local]
        ends_global = [_apply_rotation(p, source_orient_matrix, base_pos).flatten() for p in ends_local_oriented]
        all_x.extend([p[0] for p in ends_global]); all_y.extend([p[1] for p in ends_global]); all_z.extend([p[2] for p in ends_global])

    # --- Focus Point ---
    if params.get('src_type') == config.SOURCE_RING:
         focus_point_param = params.get('focus_point', config.INFINITY_SYMBOL)
         if focus_point_param != config.INFINITY_SYMBOL:
              try:
                  focus_L = float(focus_point_param)
                  cone_axis_local = np.array([0, 0, -1]); cone_axis_global = source_orient_matrix @ cone_axis_local
                  focus_point_global = base_pos + cone_axis_global * focus_L
                  all_x.append(focus_point_global[0]); all_y.append(focus_point_global[1]); all_z.append(focus_point_global[2])
              except ValueError: pass

    # --- Emission Intersection at Z=0 (Estimate for bounds) ---
    # This is a rough estimate to ensure plot limits are sufficient.
    # Actual intersection polygon is drawn in geom_emission.
    max_theta_rad = math.radians(params.get('max_theta', 30.0)) # This is half-angle
    if src_z > 0: # Only if source is above target
        # Estimate max spread at Z=0 based on emission from source edges
        if src_type == config.SOURCE_RING or src_type == config.SOURCE_CIRCULAR:
            src_radius_for_spread = params.get('src_diameter', 10.0) / 2.0
            # Consider the furthest point of the source from its center projected onto XY
            # and add the spread from there
            # This is complex with arbitrary rotations, so use a simpler, larger estimate
            max_r_source_proj = src_radius_for_spread # Approximation
            intersect_radius_est = max_r_source_proj + src_z * math.tan(max_theta_rad)
        elif src_type == config.SOURCE_LINEAR:
            src_length_for_spread = params.get('src_length', 20.0) / 2.0
            intersect_radius_est = src_length_for_spread + src_z * math.tan(max_theta_rad)
        else: # Point source
            intersect_radius_est = src_z * math.tan(max_theta_rad)

        all_x.extend([src_x - intersect_radius_est, src_x + intersect_radius_est])
        all_y.extend([src_y - intersect_radius_est, src_y + intersect_radius_est])
    all_z.append(0) # Ensure Z=0 is always included

    # Final Bounds
    min_x = min(all_x); max_x = max(all_x)
    min_y = min(all_y); max_y = max(all_y)
    min_z = min(all_z); max_z = max(all_z)

    # Add small buffer if min/max are the same to avoid zero range
    if max_x == min_x: max_x += 1; min_x -= 1
    if max_y == min_y: max_y += 1; min_y -= 1
    if max_z == min_z: max_z += 1; min_z -= 1

    return min_x, max_x, min_y, max_y, min_z, max_z


def setup_plot_limits_and_aspect(ax_top: plt.Axes, ax_side: plt.Axes, params: dict):
    """
    Calculates limits to encompass all objects and sets limits and equal aspect ratio
    for both axes, ensuring unified scaling and minimal empty space.
    """
    min_x, max_x, min_y, max_y, min_z, max_z = _get_object_bounds(params)

    # Determine the center of the bounding box of all objects
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    center_z = (min_z + max_z) / 2.0

    # Determine the largest span required to show all objects
    span_x = max_x - min_x
    span_y = max_y - min_y
    span_z = max_z - min_z
    
    # The side view (XZ) will use span_x and span_z.
    # The top view (XY) will use span_x and span_y.
    # To maintain 'equal' aspect and fit both views, we find the overall max span.
    max_span_for_equal_aspect = max(span_x, span_y, span_z)
    if max_span_for_equal_aspect < 20: max_span_for_equal_aspect = 20 # Min span

    # Add a small padding (e.g., 10% of the max_span)
    padding = max_span_for_equal_aspect * 0.10

    # Calculate limits for a square view centered around the data, using max_span + padding
    half_view_size = (max_span_for_equal_aspect / 2.0) + padding

    final_limit_x = (center_x - half_view_size, center_x + half_view_size)
    final_limit_y = (center_y - half_view_size, center_y + half_view_size)
    final_limit_z = (center_z - half_view_size, center_z + half_view_size)


    # Apply calculated limits
    ax_top.set_xlim(final_limit_x)
    ax_top.set_ylim(final_limit_y)
    ax_side.set_xlim(final_limit_x) # Use same X limits for alignment
    ax_side.set_ylim(final_limit_z)

    # Set labels, titles, grid, aspect ratio
    ax_top.set_xlabel("X (мм)")
    ax_top.set_ylabel("Y (мм)")
    ax_top.set_title("Вид сверху (XY)")
    ax_top.grid(True, linestyle=':')
    ax_top.set_aspect('equal', adjustable='box')

    ax_side.set_xlabel("X (мм)")
    ax_side.set_ylabel("Z (мм)")
    ax_side.set_title("Вид сбоку (XZ)")
    ax_side.grid(True, linestyle=':')
    ax_side.set_aspect('equal', adjustable='box')


def plot_geometry_2d(fig: plt.Figure, params: dict) -> tuple:
    """
    Main function to plot the 2D geometry projections on a given Figure.
    Returns handles and labels for the legend.
    """
    fig.clear()
    ax_top = fig.add_subplot(1, 2, 1)
    ax_side = fig.add_subplot(1, 2, 2)

    plot_target_2d(ax_top, ax_side, params)
    plot_source_2d(ax_top, ax_side, params)
    plot_emission_2d(ax_top, ax_side, params)
    setup_plot_limits_and_aspect(ax_top, ax_side, params)

    handles_top, labels_top = ax_top.get_legend_handles_labels()
    handles_side, labels_side = ax_side.get_legend_handles_labels()
    all_labels = labels_top + labels_side
    all_handles = handles_top + handles_side

    unique_labels = OrderedDict()
    label_priority = [
        'Мишень', 'Диск планеты', 'Орбита',
        'Источник', 'Направление источника',
        'Ось орбиты', 'Ось планеты',
        'Фокусировка', 'Точка фокуса L', 'Конус', 'Эмиссия', 'Область пересечения'
    ]
    temp_dict = {l:h for h,l in zip(all_handles, all_labels)}

    for base_label in label_priority:
        full_label_found = None
        if base_label in temp_dict and base_label not in unique_labels:
             unique_labels[base_label] = temp_dict[base_label]
        else:
             for l_actual in temp_dict:
                 if l_actual.startswith(base_label):
                     if l_actual not in unique_labels:
                         full_label_found = l_actual
                         break
             if full_label_found:
                 unique_labels[full_label_found] = temp_dict[full_label_found]

    for label, handle in temp_dict.items():
        if label not in unique_labels:
             unique_labels[label] = handle

    return unique_labels.values(), unique_labels.keys()

# Example usage
if __name__ == '__main__':
    try:
        from scipy.spatial import ConvexHull
        print("Scipy found, convex hull enabled for intersection area.")
    except ImportError:
        print("Scipy not found, intersection area for non-ring sources might be limited to points.")

    fig_test = plt.figure(figsize=(11, 5.5)) # Adjusted for legend below

    example_params_dome = {
        'target_type': config.TARGET_DOME, 'diameter': 100, 'dome_radius': 70,
        'src_type': config.SOURCE_POINT, 'src_x': 0, 'src_y': 0, 'src_z': 150,
        'rot_x': 0, 'rot_y': 0,
        'max_theta': 30
    }
    handles, labels = plot_geometry_2d(fig_test, example_params_dome)
    fig_test.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=4, fontsize='small')
    fig_test.subplots_adjust(left=0.06, right=0.96, bottom=0.15, top=0.9, wspace=0.2)
    plt.show()
