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
from ..core.distribution import rotation_matrix # <<< Ключевой импорт
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
            # Это может произойти, если передана неправильная форма, например (N,) вместо (3,) или (3,N)
            raise ValueError(f"Invalid shape for 1D local points: {points_local.shape}. Expected (3,).")
    elif points_local.shape[0] != 3:
        # Try transposing if shape is (N, 3)
        if points_local.shape[1] == 3:
            points_local = points_local.T
        else:
             raise ValueError(f"Invalid shape for 2D local points: {points_local.shape}. Expected (3, N) or (N, 3).")

    # Ensure base_pos is 1D (3,) before reshaping for addition
    if base_pos.shape != (3,):
         if base_pos.size == 3: # Если это, например, [[x],[y],[z]] или [x,y,z]
             base_pos = base_pos.flatten()
         else:
              raise ValueError(f"Invalid shape for base_pos: {base_pos.shape}. Expected (3,).")
    
    # Perform rotation and translation
    # orient_matrix (3, 3) @ points_local (3, N) -> result (3, N)
    # Add base_pos (3,) which broadcasts to (3, N) due to [:, np.newaxis]
    points_global = orient_matrix @ points_local + base_pos[:, np.newaxis]

    # If the original input was effectively a single point (after potential reshape to (3,1)),
    # return a 1D array (3,) for consistency if a single point was intended.
    if points_global.shape[1] == 1:
        return points_global.flatten()
    else:
        return points_global # Return as (3, N)


# --- Plotting Function ---
def plot_source_2d(ax_top: plt.Axes, ax_side: plt.Axes, params: dict):
    """Plots the source's 2D projections."""
    src_type = params.get('src_type', config.SOURCE_POINT) # Тип источника
    src_x = params.get('src_x', 0.0)    # Смещение источника по X
    src_y = params.get('src_y', 0.0)    # Смещение источника по Y
    src_z = params.get('src_z', 100.0)  # Высота источника Z
    rot_x_deg = params.get('rot_x', 0.0) # Угол наклона вокруг X в градусах
    rot_y_deg = params.get('rot_y', 0.0) # Угол наклона вокруг Y в градусах
    
    # Параметры отрисовки
    color = 'red'       # Цвет для источника
    alpha = 0.6         # Прозрачность для заливки
    linewidth = 1.5     # Толщина линий

    # --- Расчет матрицы ориентации источника ---
    # Важно: последовательность вращений (например, сначала Y, потом X)
    # должна соответствовать тому, как это интерпретируется в ядре симуляции.
    # Здесь предполагается: вращение вокруг Y, затем вокруг X глобальной системы координат.
    rot_x_rad = math.radians(rot_x_deg)
    rot_y_rad = math.radians(rot_y_deg)
    
    # Матрица поворота вокруг оси Y
    mat_rot_y = rotation_matrix(np.array([0, 1, 0]), rot_y_rad)
    # Матрица поворота вокруг оси X
    mat_rot_x = rotation_matrix(np.array([1, 0, 0]), rot_x_rad)
    # Итоговая матрица ориентации источника (применяется Y, затем X)
    source_orient_matrix = mat_rot_x @ mat_rot_y
    
    # Базовая позиция центра источника в глобальных координатах
    base_pos = np.array([src_x, src_y, src_z]) # Должен быть (3,)

    # --- Вид сверху (XY плоскость) ---
    # Рисуем центр источника
    ax_top.plot(src_x, src_y, 'o', color=color, markersize=5, label='Источник (центр)')

    if src_type == config.SOURCE_RING or src_type == config.SOURCE_CIRCULAR:
        src_diameter = params.get('src_diameter', 10.0) # Диаметр источника (кольца или круга)
        src_radius = src_diameter / 2.0
        
        # Генерируем точки на окружности в локальной системе координат источника (плоскость XY источника)
        theta_ring = np.linspace(0, 2 * np.pi, 50, endpoint=True) # 50 точек для гладкой окружности
        x_local_ring = src_radius * np.cos(theta_ring)
        y_local_ring = src_radius * np.sin(theta_ring)
        z_local_ring = np.zeros_like(theta_ring) # z=0 в локальной системе источника
        
        points_local_ring = np.vstack((x_local_ring, y_local_ring, z_local_ring)) # Форма (3, 50)
        
        # Применяем ориентацию и смещение для получения глобальных координат точек кольца/круга
        points_global_ring = _apply_rotation(points_local_ring, source_orient_matrix, base_pos) # Должно быть (3, 50)
        
        label_src_shape_top = f'Источник ({src_type})'
        if src_type == config.SOURCE_CIRCULAR:
             # Для круглого источника заливаем область
             ax_top.fill(points_global_ring[0,:], points_global_ring[1,:], color=color, alpha=alpha*0.5, label=label_src_shape_top)
             ax_top.plot(points_global_ring[0,:], points_global_ring[1,:], '-', color=color, linewidth=linewidth) # И контур
        else: # Для кольцевого источника только контур
             ax_top.plot(points_global_ring[0,:], points_global_ring[1,:], '-', color=color, linewidth=linewidth, label=label_src_shape_top)

    elif src_type == config.SOURCE_LINEAR:
        src_length = params.get('src_length', 20.0)     # Длина линейного источника
        src_angle_deg = params.get('src_angle', 0.0)    # Локальный угол поворота линии источника (вокруг его Z)
        
        # Локальные координаты концов линии (вдоль локальной оси X источника)
        # Линия от -L/2 до L/2 по локальной X
        x_local_line_ends_initial = np.array([-src_length/2, src_length/2])
        y_local_line_ends_initial = np.zeros(2)
        z_local_line_ends_initial = np.zeros(2)
        points_local_line_initial = np.vstack((x_local_line_ends_initial, y_local_line_ends_initial, z_local_line_ends_initial)) # Форма (3, 2)
        
        # Применяем локальный поворот линии источника (src_angle) вокруг локальной оси Z источника
        rot_src_angle_rad = math.radians(src_angle_deg)
        mat_rot_src_angle_local_z = rotation_matrix(np.array([0,0,1]), rot_src_angle_rad)
        points_local_line_oriented = mat_rot_src_angle_local_z @ points_local_line_initial # Форма (3, 2)
        
        # Применяем основную ориентацию источника (rot_x, rot_y) и смещение (src_x, src_y, src_z)
        points_global_line = _apply_rotation(points_local_line_oriented, source_orient_matrix, base_pos) # Должно быть (3, 2)
        
        # Рисуем линию на виде сверху (проекция XY)
        ax_top.plot(points_global_line[0,:], points_global_line[1,:], '-', color=color, linewidth=linewidth + 0.5, label='Источник (Линия)')

    # --- Вид сбоку (XZ плоскость) ---
    # Рисуем центр источника
    ax_side.plot(src_x, src_z, 'o', color=color, markersize=5, label='Источник (центр)') # Метка может дублироваться, легенда объединит

    if src_type == config.SOURCE_RING or src_type == config.SOURCE_CIRCULAR:
        # Используем те же глобальные точки, что и для вида сверху
        # points_global_ring уже рассчитаны и имеют форму (3, 50)
        # Просто берем X и Z компоненты для проекции
        label_src_shape_side = f'Источник ({src_type})' # Метка для легенды
        ax_side.plot(points_global_ring[0,:], points_global_ring[2,:], '-', color=color, linewidth=linewidth, label=label_src_shape_side)

    elif src_type == config.SOURCE_LINEAR:
        # Используем те же глобальные точки концов линии, что и для вида сверху
        # points_global_line уже рассчитаны и имеют форму (3, 2)
        # Берем X и Z компоненты для проекции
        ax_side.plot(points_global_line[0,:], points_global_line[2,:], '-', color=color, linewidth=linewidth + 0.5, label='Источник (Линия)')

    # --- Отрисовка вектора ориентации источника (стрелка) ---
    # Длина стрелки для визуализации (можно сделать зависимой от масштаба)
    arrow_visualization_length = src_z / 2.0 if src_z > 10 else 5.0
    if arrow_visualization_length <= 0 : arrow_visualization_length = 20.0 # Фоллбэк, если источник на Z=0 или ниже

    # Локальный вектор направления источника (вдоль -Z локальной оси источника)
    local_direction_vector_for_arrow = np.array([0, 0, -1.0]) * arrow_visualization_length # Форма (3,)
    
    # Применяем только ориентацию (без смещения) к вектору направления,
    # а затем добавляем смещение к начальной и конечной точке стрелки.
    # Конечная точка стрелки в глобальных координатах:
    global_direction_vector_end_point = _apply_rotation(local_direction_vector_for_arrow, source_orient_matrix, base_pos) # Должно быть (3,)
    
    arrow_label = 'Направление источника'
    head_width_val = max(1.0, arrow_visualization_length * 0.1) # Ширина шляпки стрелки
    head_length_val = max(1.5, arrow_visualization_length * 0.15) # Длина шляпки стрелки
    
    # Начальные и конечные точки для стрелки на каждом виде
    # Вид сверху (XY)
    start_arrow_top_xy = base_pos[:2] # (x, y) центра источника
    end_arrow_top_xy = global_direction_vector_end_point[:2] # (x, y) конца вектора направления
    
    # Вид сбоку (XZ)
    start_arrow_side_xz = np.array([base_pos[0], base_pos[2]]) # (x, z) центра источника
    end_arrow_side_xz = np.array([global_direction_vector_end_point[0], global_direction_vector_end_point[2]]) # (x, z) конца вектора

    # Рисуем стрелку, если ее длина больше некоторого порога (чтобы избежать ошибок Matplotlib)
    if np.linalg.norm(end_arrow_top_xy - start_arrow_top_xy) > 1e-3:
        ax_top.arrow(start_arrow_top_xy[0], start_arrow_top_xy[1], 
                     end_arrow_top_xy[0] - start_arrow_top_xy[0],  # dx
                     end_arrow_top_xy[1] - start_arrow_top_xy[1],  # dy
                     head_width=head_width_val, head_length=head_length_val,
                     fc='darkred', ec='darkred', length_includes_head=True, label=arrow_label)
    
    if np.linalg.norm(end_arrow_side_xz - start_arrow_side_xz) > 1e-3:
        ax_side.arrow(start_arrow_side_xz[0], start_arrow_side_xz[1],
                      end_arrow_side_xz[0] - start_arrow_side_xz[0],  # dx
                      end_arrow_side_xz[1] - start_arrow_side_xz[1],  # dz
                      head_width=head_width_val, head_length=head_length_val,
                      fc='darkred', ec='darkred', length_includes_head=True, label=arrow_label)
