# coating_simulator_project/coating_simulator/visualization/geom_target.py
"""
Functions for plotting the 2D projections of the target geometry.
Функции для отрисовки 2D-проекций геометрии мишени.
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Используем относительные импорты
from .. import config

def plot_target_2d(ax_top: plt.Axes, ax_side: plt.Axes, params: dict):
    """Plots the target's 2D projections."""
    target_type = params.get('target_type')
    color = 'blue'
    alpha = 0.5
    target_label = 'Мишень' # Base label

    # --- Top View (XY Plane) ---
    if target_type == config.TARGET_DISK or target_type == config.TARGET_DOME:
        diameter = params.get('diameter', 100.0)
        radius = diameter / 2.0
        label = f"{target_label} ({target_type})"
        circle_top = patches.Circle((0, 0), radius, edgecolor=color, facecolor='lightblue', alpha=alpha, label=label)
        ax_top.add_patch(circle_top)
        if radius == 0:
             ax_top.plot([],[], color='lightblue', label=label)

    elif target_type == config.TARGET_LINEAR:
        length = params.get('length', 200.0)
        width = params.get('width', 100.0)
        label = f"{target_label} (Линейная)"
        rect_top = patches.Rectangle((-length/2, -width/2), length, width, edgecolor=color, facecolor='lightblue', alpha=alpha, label=label)
        ax_top.add_patch(rect_top)
    elif target_type == config.TARGET_PLANETARY:
        orbit_diameter = params.get('orbit_diameter', 200.0)
        planet_diameter = params.get('planet_diameter', 50.0)
        orbit_radius = orbit_diameter / 2.0
        planet_radius = planet_diameter / 2.0
        orbit_path = patches.Circle((0, 0), orbit_radius, edgecolor='gray', facecolor='none', linestyle='--', label='Орбита')
        ax_top.add_patch(orbit_path)
        # Draw one planet at a representative position (e.g., 0 degrees)
        planet_center_x = orbit_radius
        planet_center_y = 0
        planet_disk = patches.Circle((planet_center_x, planet_center_y), planet_radius, edgecolor=color, facecolor='lightblue', alpha=alpha)
        ax_top.add_patch(planet_disk)
        ax_top.plot([], [], color='lightblue', marker='o', linestyle='None', markersize=5, label='Диск планеты') # For legend

    # --- Side View (XZ Plane) ---
    if target_type == config.TARGET_DISK or target_type == config.TARGET_LINEAR:
        size = params.get('diameter', params.get('length', 100.0))
        label = f"{target_label} ({'Диск' if target_type == config.TARGET_DISK else 'Линейная'})"
        ax_side.plot([-size/2, size/2], [0, 0], color=color, linewidth=2, label=label)

    elif target_type == config.TARGET_DOME:
        diameter = params.get('diameter', 100.0)
        dome_radius = params.get('dome_radius', 50.0)
        base_radius = diameter / 2.0
        label = f"{target_label} (Купол)"

        if dome_radius <= 0:
             ax_side.plot([-base_radius, base_radius], [0, 0], color=color, linewidth=2, label=f'{label} - некорр. R_купола')
             return

        x_limit = min(base_radius, dome_radius)
        x_arc = np.linspace(-x_limit, x_limit, 100)
        sqrt_arg = dome_radius**2 - x_arc**2
        sqrt_arg[sqrt_arg < 0] = 0
        z_arc = dome_radius - np.sqrt(sqrt_arg) # Peak at Z=0, opens down
        ax_side.plot(x_arc, z_arc, color=color, linewidth=2, label=label)

        if base_radius < dome_radius:
            sqrt_arg_base = dome_radius**2 - base_radius**2
            z_at_base = dome_radius - np.sqrt(sqrt_arg_base) if sqrt_arg_base >= 0 else dome_radius
            ax_side.plot([-base_radius, base_radius], [z_at_base, z_at_base], color=color, linestyle=':', linewidth=1, alpha=0.7)

    elif target_type == config.TARGET_PLANETARY:
         orbit_diameter = params.get('orbit_diameter', 200.0)
         planet_diameter = params.get('planet_diameter', 50.0)
         orbit_radius = orbit_diameter / 2.0
         planet_radius = planet_diameter / 2.0
         # Position of the single drawn planet in side view
         planet_center_x_side = orbit_radius
         # Draw planet disk as line
         ax_side.plot([planet_center_x_side - planet_radius, planet_center_x_side + planet_radius], [0, 0], color=color, linewidth=2, label='Диск планеты')

         # --- Рисуем оси вращения (будут обрезаны установленными пределами) ---
         # Используем очень большие пределы Z для рисования, фактические пределы обрежут линии
         # Важно: Рисуем их ДО установки финальных пределов в geom_layout
         z_axis_min = -1e6
         z_axis_max = 1e6
         # Ось вращения орбиты (вертикальная линия в центре X=0)
         ax_side.plot([0, 0], [z_axis_min, z_axis_max], color='gray', linestyle='-.', linewidth=1, label='Ось орбиты')
         # Ось вращения планеты (вертикальная линия через центр планеты)
         ax_side.plot([planet_center_x_side, planet_center_x_side], [z_axis_min, z_axis_max], color=color, linestyle=':', linewidth=1, label='Ось планеты')
         # ----------------------------------------------------
         # Add dummy plot for orbit diameter legend entry
         ax_side.plot([], [], color='gray', linestyle='--', label='Орбита')

