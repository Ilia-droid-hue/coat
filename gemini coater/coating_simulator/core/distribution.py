# coating_simulator_project/coating_simulator/core/distribution.py
"""
Functions for generating particle distributions and performing rotations.
Vectorized versions for improved performance with NumPy.
"""

import numpy as np
import math # <<< ИСПРАВЛЕНИЕ: Добавлен импорт math
from .. import config

def rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """
    Calculates the rotation matrix for rotating around a given axis by an angle.
    Вычисляет матрицу поворота вокруг заданной оси на заданный угол.
    """
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0:
        # Если ось нулевая, возвращаем единичную матрицу (нет вращения)
        return np.identity(3)
    axis = axis / norm # Нормализуем ось вращения
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    omc = 1.0 - c # 1 - cos(angle)
    return np.array([
        [c + x*x*omc,       x*y*omc - z*s,   x*z*omc + y*s],
        [y*x*omc + z*s,   c + y*y*omc,       y*z*omc - x*s],
        [z*x*omc - y*s,   z*y*omc + x*s,   c + z*z*omc]
    ])

# --- Векторизованные функции генерации ---

def sample_gaussian_theta_vectorized(theta_max_rad: float, sigma_rad: float, num_samples: int) -> np.ndarray:
    """
    Samples emission angles theta from a truncated Gaussian distribution (vectorized).
    Использует rejection sampling. Может быть неэффективным для больших theta_max / sigma.
    Генерирует углы тета из усеченного нормального распределения (векторизовано).
    """
    if math.isclose(theta_max_rad, 0.0):
        return np.zeros(num_samples)
    if sigma_rad <= 0:
        # print("Warning: sigma_rad <= 0 for Gaussian distribution. Returning theta=0.")
        return np.zeros(num_samples)

    # Оценочный коэффициент "пересэмплирования"
    oversampling_factor = 5
    if theta_max_rad > 1e-9 and sigma_rad > 1e-9: # Избегаем деления на ноль
        if theta_max_rad / sigma_rad < 1: # Если theta_max сильно меньше sigma, отбраковка будет частой
            oversampling_factor = 10 + int(sigma_rad / (theta_max_rad + 1e-9))
    elif sigma_rad <= 1e-9 and theta_max_rad > 1e-9: # Очень узкое распределение, почти дельта-функция
        return np.full(num_samples, theta_max_rad * np.random.rand(num_samples)) # Возвращаем случайные в пределах theta_max

    samples_needed = num_samples
    collected_samples_list = [] # Используем список для начального сбора

    attempts = 0
    max_attempts = 100 # Предотвращение бесконечного цикла

    while samples_needed > 0 and attempts < max_attempts:
        num_to_generate = int(samples_needed * oversampling_factor)
        if num_to_generate == 0 and samples_needed > 0 :
            num_to_generate = samples_needed * 2 # Ensure we generate something
        if num_to_generate == 0: break # Если все еще 0, выходим

        theta_candidates = np.abs(np.random.normal(0, sigma_rad, num_to_generate))
        valid_mask = theta_candidates <= theta_max_rad
        new_samples = theta_candidates[valid_mask]

        if new_samples.size > 0:
            collected_samples_list.append(new_samples)

        current_collected_size = sum(s.size for s in collected_samples_list)
        samples_needed = num_samples - current_collected_size
        attempts += 1

    if collected_samples_list:
        collected_samples = np.concatenate(collected_samples_list)
        if collected_samples.size >= num_samples:
            return collected_samples[:num_samples]
        elif collected_samples.size > 0:
            # print(f"Warning: Gaussian sampling only collected {collected_samples.size}/{num_samples} samples. Repeating last valid.")
            return np.pad(collected_samples, (0, num_samples - collected_samples.size), 'wrap')
    
    # print(f"Warning: Gaussian sampling failed to collect enough samples after {max_attempts} attempts. Returning uniform within theta_max.")
    return np.random.uniform(0, theta_max_rad, num_samples)


def sample_cosine_power_theta_vectorized(theta_max_rad: float, m: float, num_samples: int) -> np.ndarray:
    """
    Samples emission angles theta according to a cosine-power distribution (vectorized).
    Использует rejection sampling.
    Генерирует углы тета согласно степенному косинусному распределению (векторизовано).
    """
    if math.isclose(theta_max_rad, 0.0):
        return np.zeros(num_samples)
    if m < 0: m = 0 # Экспонента не может быть отрицательной

    samples = np.zeros(num_samples)
    count = 0
    # Увеличиваем количество попыток, чтобы избежать долгого ожидания при больших m
    max_total_attempts = num_samples * max(100, int(m * 10)) if m > 1 else num_samples * 1000
    current_attempts = 0

    while count < num_samples and current_attempts < max_total_attempts:
        # Генерируем кандидатов батчами для эффективности
        batch_size = min(num_samples - count, 10000) # Размер батча
        theta_candidates = np.random.uniform(0, theta_max_rad, batch_size)
        
        # Вычисляем вероятность принятия для каждого кандидата
        # Используем np.cos для векторизации
        cos_theta_candidates = np.cos(theta_candidates)
        # Обработка случая, когда cos(theta) может быть отрицательным (хотя theta в [0, pi/2])
        # Для theta > pi/2, cos(theta) < 0. Если m - не целое, это может вызвать проблемы.
        # Но theta_max_rad обычно <= pi/2. Если m=0, prob=1.
        if m == 0:
            probabilities = np.ones_like(cos_theta_candidates)
        else:
            probabilities = np.power(np.maximum(0, cos_theta_candidates), m) # maximum(0,...) для избежания отрицательных оснований

        # Rejection sampling
        random_uniform_values = np.random.rand(batch_size)
        accepted_mask = random_uniform_values <= probabilities
        accepted_samples = theta_candidates[accepted_mask]
        
        num_accepted = accepted_samples.size
        if count + num_accepted > num_samples:
            num_to_take = num_samples - count
            samples[count : count + num_to_take] = accepted_samples[:num_to_take]
            count += num_to_take
        else:
            samples[count : count + num_accepted] = accepted_samples
            count += num_accepted
        
        current_attempts += batch_size
    
    if count < num_samples:
        # print(f"Warning: Cosine-power sampling only collected {count}/{num_samples}. Filling with random up to theta_max.")
        remaining = num_samples - count
        samples[count:] = np.random.uniform(0, theta_max_rad, remaining)
    return samples

def sample_uniform_solid_angle_theta_vectorized(theta_max_rad: float, num_samples: int) -> np.ndarray:
    """
    Samples emission angles theta for uniform distribution over a solid angle cone (vectorized).
    Генерирует углы тета для равномерного распределения по телесному углу конуса (векторизовано).
    """
    if math.isclose(theta_max_rad, 0.0):
        return np.zeros(num_samples)
        
    cos_theta_max = math.cos(theta_max_rad)
    # Распределение для cos(theta) равномерно на [cos_theta_max, 1]
    # u = (cos_theta - cos_theta_max) / (1 - cos_theta_max)
    # cos_theta = cos_theta_max + u * (1 - cos_theta_max)
    
    u_array = np.random.rand(num_samples)
    cos_theta_array = cos_theta_max + u_array * (1.0 - cos_theta_max)
    # Клиппинг на всякий случай из-за ошибок округления
    cos_theta_array = np.clip(cos_theta_array, -1.0, 1.0)
    return np.arccos(cos_theta_array)

def sample_emission_vector_vectorized(dist_type: str, theta_max_deg: float, params: dict, num_samples: int) -> np.ndarray:
    """
    Generates an array of random emission direction vectors (dx, dy, dz) in the local source coordinates.
    The source is assumed to be oriented along the -Z axis locally.
    Возвращает массив случайных векторов направления эмиссии (dx, dy, dz) в локальной системе координат источника.
    Источник локально ориентирован вдоль оси -Z.
    Shape of returned array: (3, num_samples)
    """
    theta_max_rad = math.radians(theta_max_deg)
    
    if math.isclose(theta_max_rad, 0.0): # Если максимальный угол 0, все частицы летят по нормали
        theta_array = np.zeros(num_samples)
    else:
        if dist_type == config.DIST_GAUSSIAN:
            sigma_rad = math.radians(params.get('sigma', 1.0)) # sigma в градусах из параметров
            theta_array = sample_gaussian_theta_vectorized(theta_max_rad, sigma_rad, num_samples)
        elif dist_type == config.DIST_COSINE_POWER:
            m = params.get('m_exp', 1.0)
            theta_array = sample_cosine_power_theta_vectorized(theta_max_rad, m, num_samples)
        elif dist_type == config.DIST_UNIFORM_SOLID:
            theta_array = sample_uniform_solid_angle_theta_vectorized(theta_max_rad, num_samples)
        else:
            # print(f"Warning: Unknown distribution type '{dist_type}'. Falling back to Uniform Solid Angle.")
            theta_array = sample_uniform_solid_angle_theta_vectorized(theta_max_rad, num_samples)

    phi_array = np.random.uniform(0, 2 * math.pi, num_samples) # Азимутальный угол фи

    # Компоненты вектора в сферических координатах (физическое определение: тета от -Z)
    # dz направлен вдоль -Z, если тета = 0
    sin_theta_array = np.sin(theta_array)
    cos_theta_array = np.cos(theta_array)
    
    dx_array = sin_theta_array * np.cos(phi_array)
    dy_array = sin_theta_array * np.sin(phi_array)
    dz_array = -cos_theta_array # Минус, так как локальная ось Z источника направлена "вниз"

    return np.vstack((dx_array, dy_array, dz_array)) # Shape (3, num_samples)

def sample_source_position_vectorized(src_type: str, params: dict, num_samples: int) -> np.ndarray:
    """
    Generates an array of random starting positions on the source surface (in local source coordinates).
    Возвращает массив случайных начальных позиций на поверхности источника (в локальной системе координат источника).
    Shape of returned array: (3, num_samples)
    """
    if src_type == config.SOURCE_POINT:
        return np.zeros((3, num_samples)) # Все частицы из точки (0,0,0) локально

    elif src_type == config.SOURCE_RING:
        src_radius = params.get('src_diameter', 0.0) / 2.0
        if src_radius <= 0: return np.zeros((3, num_samples)) # Если радиус 0, считаем точечным
        
        phi_array = np.random.uniform(0, 2 * math.pi, num_samples)
        x_array = src_radius * np.cos(phi_array)
        y_array = src_radius * np.sin(phi_array)
        z_array = np.zeros(num_samples) # Кольцо в плоскости XY источника
        return np.vstack((x_array, y_array, z_array))

    elif src_type == config.SOURCE_CIRCULAR:
        src_radius = params.get('src_diameter', 0.0) / 2.0
        if src_radius <= 0: return np.zeros((3, num_samples))
        
        # Для равномерного распределения по площади диска, r нужно генерировать из sqrt(uniform * R^2)
        r_squared_array = np.random.uniform(0, src_radius**2, num_samples)
        r_array = np.sqrt(r_squared_array)
        phi_array = np.random.uniform(0, 2 * math.pi, num_samples)
        
        x_array = r_array * np.cos(phi_array)
        y_array = r_array * np.sin(phi_array)
        z_array = np.zeros(num_samples) # Диск в плоскости XY источника
        return np.vstack((x_array, y_array, z_array))

    elif src_type == config.SOURCE_LINEAR:
        src_length = params.get('src_length', 0.0)
        src_angle_deg = params.get('src_angle', 0.0) # Локальный угол поворота линии источника
        if src_length <= 0: return np.zeros((3, num_samples))

        # Генерируем точки вдоль локальной оси X источника (от -L/2 до L/2)
        t_array = np.random.uniform(-src_length / 2.0, src_length / 2.0, num_samples)
        # Изначально линия вдоль оси X источника
        local_pos_array = np.vstack((t_array, np.zeros(num_samples), np.zeros(num_samples)))

        # Если задан локальный угол поворота линии источника (src_angle)
        if not math.isclose(src_angle_deg, 0.0):
            rot_angle_rad = math.radians(src_angle_deg)
            # Вращение вокруг локальной оси Z источника
            rot_mat_local_z = rotation_matrix(np.array([0, 0, 1]), rot_angle_rad)
            return rot_mat_local_z @ local_pos_array # (3,3) @ (3,N) -> (3,N)
        else:
            return local_pos_array # Без локального поворота
    else:
        # print(f"Warning: Unknown source type '{src_type}'. Treating as point source.")
        return np.zeros((3, num_samples))

# --- Старые одночастичные функции (можно оставить для справки или удалить) ---
# Эти функции больше не используются в основном цикле симуляции,
# который теперь работает с векторизованными версиями.

def sample_gaussian_theta(theta_max_rad: float, sigma_rad: float) -> float:
    """Генерирует один угол тета из усеченного нормального распределения."""
    if math.isclose(theta_max_rad, 0.0): return 0.0
    if sigma_rad <= 0: return 0.0 # Возвращаем 0, если sigma некорректна
    # Rejection sampling
    attempts = 0
    max_attempts = 1000 # Предотвращение зависания
    while attempts < max_attempts:
        theta = abs(np.random.normal(0, sigma_rad))
        if theta <= theta_max_rad:
            return theta
        attempts += 1
    return np.random.uniform(0, theta_max_rad) # Фоллбэк, если не удалось сгенерировать

def sample_cosine_power_theta(theta_max_rad: float, m: float) -> float:
    """Генерирует один угол тета согласно степенному косинусному распределению."""
    if math.isclose(theta_max_rad, 0.0): return 0.0
    if m < 0: m = 0
    attempts = 0
    max_attempts = 1000 * max(1, int(m+1)) # Больше попыток для больших m
    while attempts < max_attempts:
        theta = np.random.uniform(0, theta_max_rad)
        try:
            prob = (math.cos(theta))**m
        except ValueError: # math domain error for cos(theta) if theta > pi/2
            prob = 0.0
        if np.random.rand() <= prob:
            return theta
        attempts +=1
    return np.random.uniform(0, theta_max_rad) # Фоллбэк

def sample_uniform_solid_angle_theta(theta_max_rad: float) -> float:
    """Генерирует один угол тета для равномерного распределения по телесному углу конуса."""
    if math.isclose(theta_max_rad, 0.0): return 0.0
    cos_theta_max = math.cos(theta_max_rad)
    # cos_theta распределен равномерно на [cos_theta_max, 1]
    u = np.random.rand()
    cos_theta = cos_theta_max + u * (1.0 - cos_theta_max)
    cos_theta = max(-1.0, min(1.0, cos_theta)) # Клиппинг для безопасности
    return math.acos(cos_theta)

def sample_emission_vector(dist_type: str, theta_max_deg: float, params: dict) -> tuple[float, float, float]:
    """
    Генерирует один случайный вектор направления эмиссии.
    (Обертка над векторизованной версией для совместимости или тестирования)
    """
    res_array = sample_emission_vector_vectorized(dist_type, theta_max_deg, params, 1)
    return res_array[0,0], res_array[1,0], res_array[2,0]

def sample_source_position(src_type: str, params: dict) -> np.ndarray:
    """
    Генерирует одну случайную начальную позицию на поверхности источника.
    (Обертка над векторизованной версией)
    """
    res_array = sample_source_position_vectorized(src_type, params, 1)
    return res_array[:,0] # Возвращает 1D массив (3,)
