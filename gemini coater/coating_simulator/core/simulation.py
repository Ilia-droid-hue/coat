# coating_simulator_project/coating_simulator/core/simulation.py
"""
Core simulation functions for different target types using Monte Carlo method.
Основные функции симуляции для различных типов мишеней методом Монте-Карло.
"""

import numpy as np
import math
import time # Для отладки времени выполнения

# Используем относительные импорты
from .distribution import rotation_matrix, sample_source_position, sample_emission_vector
from .. import config # Импортируем константы

def _check_intersection_disk(point: np.ndarray, radius: float) -> bool:
    """Checks if a point intersects with a flat disk target."""
    px, py, pz = point
    # Check if within radius and close to Z=0 plane
    # Проверяем, находится ли точка в пределах радиуса и близко к плоскости Z=0
    return np.hypot(px, py) <= radius and abs(pz) < config.SIM_INTERSECTION_TOLERANCE

def _check_intersection_dome(point: np.ndarray, diameter: float, dome_radius: float) -> bool:
    """Checks if a point intersects with a dome target."""
    px, py, pz = point
    radius = diameter / 2.0
    r_sq = px**2 + py**2

    if r_sq > radius**2: # Outside the base diameter
        return False

    # Calculate expected Z on the dome surface (center at 0,0,-dome_radius)
    # Вычисляем ожидаемый Z на поверхности купола (центр в 0,0,-dome_radius)
    # Equation: x^2 + y^2 + (z + dome_radius)^2 = dome_radius^2
    # z = sqrt(dome_radius^2 - x^2 - y^2) - dome_radius
    # We only consider the upper surface, so z should be negative or zero.
    # Мы рассматриваем только верхнюю поверхность, поэтому z должен быть отрицательным или нулевым.
    if dome_radius**2 < r_sq: # Avoid sqrt domain error if r > dome_radius (should be caught by radius check)
         return False
    z_expected = math.sqrt(dome_radius**2 - r_sq) - dome_radius

    # Check if the point's Z is close to the expected Z on the dome
    # Проверяем, близок ли Z точки к ожидаемому Z на куполе
    return abs(pz - z_expected) < config.SIM_INTERSECTION_TOLERANCE

def _check_intersection_linear(point: np.ndarray, length: float, width: float) -> bool:
    """Checks if a point intersects with a flat linear target."""
    px, py, pz = point
    half_length = length / 2.0
    half_width = width / 2.0
    # Check if within bounds and close to Z=0 plane
    # Проверяем, находится ли точка в границах и близко к плоскости Z=0
    return (abs(pz) < config.SIM_INTERSECTION_TOLERANCE and
            -half_length <= px <= half_length and
            -half_width <= py <= half_width)

def _check_intersection_planetary_disk(point: np.ndarray, disk_radius: float) -> bool:
    """Checks if a point intersects with a planetary disk (assumed flat at Z=0 in its own frame)."""
    # In the planetary simulation, the point 'p' is already transformed relative to the disk center.
    # В симуляции планетарного движения точка 'p' уже преобразована относительно центра диска.
    px, py, pz = point
    # Check if within radius and close to Z=0 plane relative to the disk center
    # Проверяем, находится ли точка в пределах радиуса и близко к плоскости Z=0 относительно центра диска
    return np.hypot(px, py) <= disk_radius and abs(pz) < config.SIM_INTERSECTION_TOLERANCE


def _run_simulation_loop(params: dict, calculate_transforms, check_intersection, progress_callback=None, cancel_event=None):
    """
    Generic Monte Carlo simulation loop.
    Общий цикл симуляции Монте-Карло.

    Args:
        params (dict): Dictionary containing all simulation parameters.
                       Словарь, содержащий все параметры симуляции.
        calculate_transforms (callable): Function to calculate time-dependent transforms (source position, rotation matrices).
                                         Функция для расчета зависящих от времени преобразований (положение источника, матрицы поворота).
        check_intersection (callable): Function to check if a point intersects the target.
                                       Функция для проверки пересечения точки с мишенью.
        progress_callback (callable, optional): Function to report progress (0-100). Defaults to None.
                                                Функция для сообщения о прогрессе (0-100). По умолчанию None.
        cancel_event (threading.Event, optional): Event to signal cancellation. Defaults to None.
                                                  Событие для сигнализации отмены. По умолчанию None.

    Returns:
        tuple: coverage_map, x_coords, y_coords, radius_grid
               Карта покрытия, координаты x, координаты y, сетка радиусов
    """
    n_particles = params['particles']
    grid_size = config.SIM_GRID_SIZE
    target_size = params.get('diameter', params.get('length', params.get('planet_diameter', 1.0))) # Determine relevant size
                                                                                                   # Определяем релевантный размер
    radius = target_size / 2.0 # Used for grid setup, actual check uses specific dimensions
                               # Используется для настройки сетки, фактическая проверка использует конкретные размеры

    # Setup grid
    # Настройка сетки
    x_coords = np.linspace(-radius, radius, grid_size)
    y_coords = np.linspace(-radius, radius, grid_size)
    xx, yy = np.meshgrid(x_coords, y_coords)
    radius_grid = np.hypot(xx, yy)
    coverage_map = np.zeros_like(xx, dtype=int) # Use integers for counts

    # Progress reporting setup
    # Настройка отчета о прогрессе
    report_interval = max(1, int(n_particles * config.SIM_PROGRESS_INTERVAL_PERCENT / 100.0))
    last_report_time = time.time()

    # --- Main Simulation Loop ---
    for i in range(n_particles):
        if cancel_event and cancel_event.is_set():
            print("Simulation cancelled.")
            break

        # 1. Calculate time-dependent transforms for this particle
        # 1. Рассчитываем зависящие от времени преобразования для этой частицы
        t = params['time'] * i / n_particles
        src_pos_base, source_rotation_matrix, target_rotation_matrix_inv = calculate_transforms(t, params)

        # 2. Sample starting position on the source (local coordinates)
        # 2. Генерируем начальную позицию на источнике (локальные координаты)
        local_src_offset = sample_source_position(params['src_type'], params)

        # 3. Transform local source position to global coordinates
        # 3. Преобразуем локальную позицию источника в глобальные координаты
        # Apply source rotation first, then add base position
        # Сначала применяем вращение источника, затем добавляем базовую позицию
        global_src_pos = src_pos_base + source_rotation_matrix.dot(local_src_offset)

        # 4. Sample emission direction vector (local to source, pointing -Z)
        # 4. Генерируем вектор направления эмиссии (локально к источнику, направлен по -Z)
        local_dir_vec = np.array(sample_emission_vector(params['dist_type'], params['max_theta'], params))

        # 5. Transform emission vector to global coordinates
        # 5. Преобразуем вектор эмиссии в глобальные координаты
        # Apply source rotation to the local direction vector
        # Применяем вращение источника к локальному вектору направления
        global_dir_vec = source_rotation_matrix.dot(local_dir_vec)

        # 6. Ray tracing: Find intersection with target plane/surface
        # 6. Трассировка лучей: Находим пересечение с плоскостью/поверхностью мишени
        # Simplified approach: step along the ray and check intersection
        # Упрощенный подход: шагаем вдоль луча и проверяем пересечение
        intersected = False
        for step in np.linspace(0, config.SIM_TRACE_MAX_DIST, config.SIM_TRACE_STEPS):
            # Current point in global coordinates
            # Текущая точка в глобальных координатах
            current_point_global = global_src_pos + global_dir_vec * step

            # Transform point to target's coordinate system (if target moves/rotates)
            # Преобразуем точку в систему координат мишени (если мишень движется/вращается)
            point_in_target_frame = target_rotation_matrix_inv.dot(current_point_global)

            # Check for intersection using the provided function
            # Проверяем пересечение с помощью предоставленной функции
            if check_intersection(point_in_target_frame, params):
                # 7. If intersected, find grid cell and increment coverage
                # 7. Если пересеклись, находим ячейку сетки и увеличиваем покрытие
                px, py, _ = point_in_target_frame # Use coordinates relative to target center

                # Find the closest grid indices (more robust than searchsorted for float precision)
                # Находим ближайшие индексы сетки (более надежно, чем searchsorted для точности float)
                ix = np.argmin(np.abs(x_coords - px))
                iy = np.argmin(np.abs(y_coords - py))

                # Check if indices are within bounds (should be if point is within radius)
                # Проверяем, находятся ли индексы в границах (должны быть, если точка в пределах радиуса)
                if 0 <= ix < grid_size and 0 <= iy < grid_size:
                     # Check if the point is actually within the cell boundaries represented by the index
                     # Проверяем, действительно ли точка находится в границах ячейки, представленной индексом
                     dx = x_coords[1] - x_coords[0] if grid_size > 1 else radius * 2
                     dy = y_coords[1] - y_coords[0] if grid_size > 1 else radius * 2
                     if (x_coords[ix] - dx/2 <= px < x_coords[ix] + dx/2 and
                         y_coords[iy] - dy/2 <= py < y_coords[iy] + dy/2):
                            coverage_map[iy, ix] += 1
                intersected = True
                break # Stop ray tracing once intersection is found

        # --- Progress Reporting ---
        if progress_callback and (i % report_interval == 0 or i == n_particles - 1):
            current_time = time.time()
            if current_time - last_report_time > 0.5 or i == n_particles - 1: # Limit updates to ~2 per second
                progress = int((i + 1) / n_particles * 100)
                progress_callback(progress)
                last_report_time = current_time

    return coverage_map, x_coords, y_coords, radius_grid


# --- Specific Simulation Setups ---

def simulate_coating_disk_dome(params: dict, progress_callback=None, cancel_event=None):
    """Simulation for rotating disk or dome targets."""
    target_type = params['target_type']
    diameter = params['diameter']
    radius = diameter / 2.0

    def calculate_transforms(t, p):
        # Target rotation
        # Вращение мишени
        omega = 2 * math.pi * p['rpm'] / 60.0
        target_angle = omega * t
        # Inverse rotation to bring global point to target frame
        # Обратное вращение для перевода глобальной точки в систему координат мишени
        target_rot_inv = rotation_matrix([0, 0, 1], -target_angle)

        # Source position and rotation
        # Положение и вращение источника
        src_base = np.array([p['src_x'], p['src_y'], p['src_z']]) # Source base position is fixed relative to world
                                                                  # Базовое положение источника фиксировано относительно мира
        # Source orientation (combine rotations around X and Y)
        # Ориентация источника (комбинируем вращения вокруг X и Y)
        rot_x_rad = math.radians(p['rot_x'])
        rot_y_rad = math.radians(p['rot_y'])
        # Apply Y rotation first, then X rotation (or vice versa, be consistent)
        # Сначала применяем вращение Y, затем вращение X (или наоборот, будьте последовательны)
        mat_rot_y = rotation_matrix([0, 1, 0], rot_y_rad)
        mat_rot_x = rotation_matrix([1, 0, 0], rot_x_rad)
        source_rot = mat_rot_x @ mat_rot_y # Combined rotation matrix

        return src_base, source_rot, target_rot_inv

    def check_intersection(point_in_target_frame, p):
        if target_type == config.TARGET_DOME:
            return _check_intersection_dome(point_in_target_frame, p['diameter'], p['dome_radius'])
        else: # Default to disk
            return _check_intersection_disk(point_in_target_frame, radius)

    # Update params with specific defaults if missing
    params.setdefault('rpm', config.DEFAULT_PROCESSING_PARAMS['rpm'])

    return _run_simulation_loop(params, calculate_transforms, check_intersection, progress_callback, cancel_event)


def simulate_linear_movement(params: dict, progress_callback=None, cancel_event=None):
    """Simulation for linearly moving flat target."""
    length = params['length']
    width = params['width']

    def calculate_transforms(t, p):
        # Target does not rotate relative to its own frame, so inverse is identity
        # Мишень не вращается относительно своей системы координат, поэтому обратное преобразование - единичная матрица
        target_rot_inv = np.identity(3)

        # Source position: Base is fixed, movement handled by target frame check
        # Положение источника: База фиксирована, движение обрабатывается проверкой в системе координат мишени
        src_base = np.array([p['src_x'], p['src_y'], p['src_z']])

        # Source orientation
        # Ориентация источника
        rot_x_rad = math.radians(p['rot_x'])
        rot_y_rad = math.radians(p['rot_y'])
        mat_rot_y = rotation_matrix([0, 1, 0], rot_y_rad)
        mat_rot_x = rotation_matrix([1, 0, 0], rot_x_rad)
        source_rot = mat_rot_x @ mat_rot_y

        # Linear movement: Calculate target offset at time t
        # Линейное движение: Рассчитываем смещение мишени в момент времени t
        speed = p['speed']
        # Movement along X axis, repeating every 'length' distance
        # Движение вдоль оси X, повторяющееся каждые 'length'
        # Position oscillates between -length/2 and +length/2
        # Позиция колеблется между -length/2 и +length/2
        total_dist = speed * t
        # Normalize distance within one cycle [0, 2*length)
        # Нормализуем расстояние в пределах одного цикла [0, 2*length)
        cycle_dist = total_dist % (2 * length)
        if cycle_dist < length:
             # Moving from -L/2 to +L/2
             # Движемся от -L/2 к +L/2
             target_offset_x = -length / 2.0 + cycle_dist
        else:
             # Moving from +L/2 to -L/2
             # Движемся от +L/2 к -L/2
             target_offset_x = length / 2.0 - (cycle_dist - length)

        # We need to transform the global point into the *moving* target frame.
        # This means subtracting the target's offset from the global point.
        # The `target_rot_inv` will include this translation.
        # Нам нужно преобразовать глобальную точку в *движущуюся* систему координат мишени.
        # Это означает вычитание смещения мишени из глобальной точки.
        # `target_rot_inv` будет включать это смещение.

        # Create a translation matrix for the inverse transform
        # Создаем матрицу сдвига для обратного преобразования
        # Since target_rot_inv is applied as M*point, we need a 4x4 matrix approach or adjust the point before check_intersection
        # Так как target_rot_inv применяется как M*point, нам нужен подход с матрицами 4x4 или корректировка точки перед check_intersection
        # Simpler: Adjust the point inside check_intersection or modify the check function signature.
        # Проще: Скорректировать точку внутри check_intersection или изменить сигнатуру функции проверки.
        # Let's pass the offset to the check function.

        # Return the offset instead of target_rot_inv for this type
        # Возвращаем смещение вместо target_rot_inv для этого типа
        target_offset = np.array([target_offset_x, 0.0, 0.0])

        # Revisit: The _run_simulation_loop expects target_rotation_matrix_inv.
        # We need a way to handle the translation.
        # Пересмотр: _run_simulation_loop ожидает target_rotation_matrix_inv.
        # Нам нужен способ обработки смещения.
        # Option 1: Modify loop to handle translation separately.
        # Option 2: Use 4x4 matrices (more complex).
        # Option 3: Adjust the point *before* calling check_intersection.

        # Let's go with Option 3 for now. The check function will implicitly handle the frame.
        # Пока выберем Вариант 3. Функция проверки неявно обработает систему координат.
        # The check_intersection_linear assumes the point is already in the target's *static* frame.
        # check_intersection_linear предполагает, что точка уже находится в *статической* системе координат мишени.
        # So, point_in_target_frame = point_global - target_offset

        # We still need to return *something* for target_rotation_matrix_inv. Identity is fine.
        # Нам все еще нужно вернуть *что-то* для target_rotation_matrix_inv. Единичная матрица подойдет.

        return src_base, source_rot, np.identity(3), target_offset # Return offset separately


    def check_intersection_moving(point_global, p, target_offset_at_t):
         # Transform global point to the target's frame at this instant
         # Преобразуем глобальную точку в систему координат мишени в данный момент
         point_in_target_frame = point_global - target_offset_at_t
         return _check_intersection_linear(point_in_target_frame, p['length'], p['width'])

    # Modify the simulation loop call slightly to handle the extra offset argument
    # Немного изменим вызов цикла симуляции для обработки дополнительного аргумента смещения

    # --- Modified Simulation Loop Call ---
    n_particles = params['particles']
    grid_size = config.SIM_GRID_SIZE
    radius = params['length'] / 2.0 # Use length for grid setup

    x_coords = np.linspace(-radius, radius, grid_size)
    y_coords = np.linspace(-params['width'] / 2.0, params['width'] / 2.0, grid_size) # Adjust Y grid for width
                                                                                      # Корректируем сетку Y по ширине
    xx, yy = np.meshgrid(x_coords, y_coords)
    radius_grid = np.hypot(xx, yy) # Note: rr is less meaningful here
                                   # Примечание: rr здесь менее значим
    coverage_map = np.zeros_like(xx, dtype=int)

    report_interval = max(1, int(n_particles * config.SIM_PROGRESS_INTERVAL_PERCENT / 100.0))
    last_report_time = time.time()

    for i in range(n_particles):
        if cancel_event and cancel_event.is_set():
            print("Simulation cancelled.")
            break

        t = params['time'] * i / n_particles
        src_pos_base, source_rotation_matrix, _, target_offset = calculate_transforms(t, params) # Get offset

        local_src_offset = sample_source_position(params['src_type'], params)
        global_src_pos = src_pos_base + source_rotation_matrix.dot(local_src_offset)

        local_dir_vec = np.array(sample_emission_vector(params['dist_type'], params['max_theta'], params))
        global_dir_vec = source_rotation_matrix.dot(local_dir_vec)

        intersected = False
        for step in np.linspace(0, config.SIM_TRACE_MAX_DIST, config.SIM_TRACE_STEPS):
            current_point_global = global_src_pos + global_dir_vec * step

            # Check intersection using the moving frame logic
            # Проверяем пересечение, используя логику движущейся системы координат
            if check_intersection_moving(current_point_global, params, target_offset):
                # Point relative to target center *at this time t*
                # Точка относительно центра мишени *в данный момент времени t*
                point_in_target_frame = current_point_global - target_offset
                px, py, _ = point_in_target_frame

                ix = np.argmin(np.abs(x_coords - px))
                iy = np.argmin(np.abs(y_coords - py))

                if 0 <= ix < grid_size and 0 <= iy < grid_size:
                     dx = x_coords[1] - x_coords[0] if grid_size > 1 else length
                     dy = y_coords[1] - y_coords[0] if grid_size > 1 else params['width']
                     if (x_coords[ix] - dx/2 <= px < x_coords[ix] + dx/2 and
                         y_coords[iy] - dy/2 <= py < y_coords[iy] + dy/2):
                            coverage_map[iy, ix] += 1
                intersected = True
                break

        if progress_callback and (i % report_interval == 0 or i == n_particles - 1):
             current_time = time.time()
             if current_time - last_report_time > 0.5 or i == n_particles - 1:
                 progress = int((i + 1) / n_particles * 100)
                 progress_callback(progress)
                 last_report_time = current_time

    return coverage_map, x_coords, y_coords, radius_grid # Return adjusted y_coords


def simulate_planetary(params: dict, progress_callback=None, cancel_event=None):
    """Simulation for planetary target motion."""
    disk_diameter = params['planet_diameter']
    disk_radius = disk_diameter / 2.0
    orbital_radius = params['orbit_diameter'] / 2.0

    def calculate_transforms(t, p):
        # Orbital and self-rotation speeds
        # Скорости орбитального и собственного вращения
        omega_orb = 2 * math.pi * p['rpm_orbit'] / 60.0
        omega_self = 2 * math.pi * p['rpm_disk'] / 60.0

        # Angles at time t
        # Углы в момент времени t
        angle_orbit = omega_orb * t
        angle_self = omega_self * t # Disk's own rotation

        # Target center position due to orbital motion
        # Положение центра мишени из-за орбитального движения
        target_center_x = orbital_radius * math.cos(angle_orbit)
        target_center_y = orbital_radius * math.sin(angle_orbit)
        target_center = np.array([target_center_x, target_center_y, 0.0])

        # Inverse transform matrix to bring global point to target frame:
        # 1. Translate by -target_center
        # 2. Rotate by -angle_self around Z axis
        # Матрица обратного преобразования для перевода глобальной точки в систему координат мишени:
        # 1. Сдвиг на -target_center
        # 2. Поворот на -angle_self вокруг оси Z
        # Again, handling translation within rotation matrix requires 4x4 or separate step.
        # Снова, обработка сдвига в матрице поворота требует 4x4 или отдельного шага.
        # Let's pass center and self_angle separately to the check function / loop.
        # Передадим центр и собственный угол отдельно в функцию проверки / цикл.

        # Source position and rotation (same as disk/dome case)
        # Положение и вращение источника (так же, как в случае диска/купола)
        src_base = np.array([p['src_x'], p['src_y'], p['src_z']])
        rot_x_rad = math.radians(p['rot_x'])
        rot_y_rad = math.radians(p['rot_y'])
        mat_rot_y = rotation_matrix([0, 1, 0], rot_y_rad)
        mat_rot_x = rotation_matrix([1, 0, 0], rot_x_rad)
        source_rot = mat_rot_x @ mat_rot_y

        # Return components needed for transformation within the loop
        # Возвращаем компоненты, необходимые для преобразования внутри цикла
        return src_base, source_rot, target_center, angle_self

    # --- Modified Simulation Loop Call for Planetary ---
    n_particles = params['particles']
    grid_size = config.SIM_GRID_SIZE
    radius = disk_radius # Grid based on planet disk size

    x_coords = np.linspace(-radius, radius, grid_size)
    y_coords = np.linspace(-radius, radius, grid_size)
    xx, yy = np.meshgrid(x_coords, y_coords)
    radius_grid = np.hypot(xx, yy)
    coverage_map = np.zeros_like(xx, dtype=int)

    report_interval = max(1, int(n_particles * config.SIM_PROGRESS_INTERVAL_PERCENT / 100.0))
    last_report_time = time.time()

    for i in range(n_particles):
        if cancel_event and cancel_event.is_set():
            print("Simulation cancelled.")
            break

        t = params['time'] * i / n_particles
        src_pos_base, source_rotation_matrix, target_center_at_t, target_self_angle_at_t = calculate_transforms(t, params)

        local_src_offset = sample_source_position(params['src_type'], params)
        global_src_pos = src_pos_base + source_rotation_matrix.dot(local_src_offset)

        local_dir_vec = np.array(sample_emission_vector(params['dist_type'], params['max_theta'], params))
        global_dir_vec = source_rotation_matrix.dot(local_dir_vec)

        intersected = False
        for step in np.linspace(0, config.SIM_TRACE_MAX_DIST, config.SIM_TRACE_STEPS):
            current_point_global = global_src_pos + global_dir_vec * step

            # Transform point to the planetary disk's frame
            # 1. Translate relative to the moving center
            # 2. Rotate opposite to the disk's self-rotation
            # Преобразуем точку в систему координат планетарного диска
            # 1. Сдвигаем относительно движущегося центра
            # 2. Вращаем в направлении, противоположном собственному вращению диска
            point_relative_to_center = current_point_global - target_center_at_t
            inv_self_rot_matrix = rotation_matrix([0, 0, 1], -target_self_angle_at_t)
            point_in_target_frame = inv_self_rot_matrix.dot(point_relative_to_center)

            # Check intersection with the disk in its own frame
            # Проверяем пересечение с диском в его собственной системе координат
            if _check_intersection_planetary_disk(point_in_target_frame, disk_radius):
                px, py, _ = point_in_target_frame

                ix = np.argmin(np.abs(x_coords - px))
                iy = np.argmin(np.abs(y_coords - py))

                if 0 <= ix < grid_size and 0 <= iy < grid_size:
                     dx = x_coords[1] - x_coords[0] if grid_size > 1 else radius * 2
                     dy = y_coords[1] - y_coords[0] if grid_size > 1 else radius * 2
                     if (x_coords[ix] - dx/2 <= px < x_coords[ix] + dx/2 and
                         y_coords[iy] - dy/2 <= py < y_coords[iy] + dy/2):
                            coverage_map[iy, ix] += 1
                intersected = True
                break

        if progress_callback and (i % report_interval == 0 or i == n_particles - 1):
             current_time = time.time()
             if current_time - last_report_time > 0.5 or i == n_particles - 1:
                 progress = int((i + 1) / n_particles * 100)
                 progress_callback(progress)
                 last_report_time = current_time

    return coverage_map, x_coords, y_coords, radius_grid
