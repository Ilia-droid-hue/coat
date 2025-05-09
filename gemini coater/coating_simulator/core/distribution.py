# coating_simulator_project/coating_simulator/core/distribution.py
"""
Functions for generating particle distributions and performing rotations.
Vectorized versions for improved performance with NumPy.
"""

import numpy as np
import math
from .. import config

def rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """
    Calculates the rotation matrix for rotating around a given axis by an angle.
    Вычисляет матрицу поворота вокруг заданной оси на заданный угол.
    """
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0:
        return np.identity(3)
    axis = axis / norm
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    omc = 1.0 - c
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
    """
    if math.isclose(theta_max_rad, 0.0):
        return np.zeros(num_samples)
    if sigma_rad <= 0:
        # print("Warning: sigma_rad <= 0 for Gaussian distribution. Returning theta=0.")
        return np.zeros(num_samples)

    # Для векторизации rejection sampling, мы генерируем больше сэмплов, чем нужно,
    # а затем отбираем подходящие. Это может быть не самым эффективным способом,
    # если вероятность отбраковки высока.
    # Оценочный коэффициент "пересэмплирования"
    oversampling_factor = 5 # Можно подбирать
    if theta_max_rad / sigma_rad < 1: # Если theta_max сильно меньше sigma, отбраковка будет частой
        oversampling_factor = 10 + int(sigma_rad / (theta_max_rad + 1e-9))


    samples_needed = num_samples
    collected_samples = np.array([])

    attempts = 0
    max_attempts = 100 # Предотвращение бесконечного цикла

    while samples_needed > 0 and attempts < max_attempts:
        num_to_generate = int(samples_needed * oversampling_factor)
        if num_to_generate == 0 and samples_needed > 0 : num_to_generate = samples_needed * 2 # Ensure we generate something
        
        theta_candidates = np.abs(np.random.normal(0, sigma_rad, num_to_generate))
        valid_mask = theta_candidates <= theta_max_rad
        new_samples = theta_candidates[valid_mask]
        
        if collected_samples.size == 0:
            collected_samples = new_samples
        else:
            collected_samples = np.concatenate((collected_samples, new_samples))
        
        samples_needed = num_samples - collected_samples.size
        attempts += 1
    
    if collected_samples.size >= num_samples:
        return collected_samples[:num_samples]
    elif collected_samples.size > 0: # Если не набрали достаточно, но что-то есть
        # print(f"Warning: Gaussian sampling only collected {collected_samples.size}/{num_samples} samples. Repeating last valid.")
        return np.pad(collected_samples, (0, num_samples - collected_samples.size), 'wrap')
    else: # Если совсем ничего не сгенерировали (маловероятно при правильных параметрах)
        # print(f"Warning: Gaussian sampling failed to collect any samples after {max_attempts} attempts. Returning uniform within theta_max.")
        return np.random.uniform(0, theta_max_rad, num_samples)


def sample_cosine_power_theta_vectorized(theta_max_rad: float, m: float, num_samples: int) -> np.ndarray:
    """
    Samples emission angles theta according to a cosine-power distribution (vectorized).
    Использует rejection sampling.
    """
    if math.isclose(theta_max_rad, 0.0):
        return np.zeros(num_samples)
    if m < 0: m = 0

    samples = np.zeros(num_samples)
    count = 0
    attempts = 0
    max_attempts_per_sample = 1000 # Чтобы избежать зависания, если m очень большое

    while count < num_samples and attempts < num_samples * max_attempts_per_sample :
        theta_candidate = np.random.uniform(0, theta_max_rad)
        try:
            prob = (math.cos(theta_candidate))**m
        except ValueError: prob = 0.0
        
        if np.random.rand() <= prob:
            samples[count] = theta_candidate
            count += 1
        attempts +=1
    
    if count < num_samples:
        # print(f"Warning: Cosine-power sampling only collected {count}/{num_samples}. Filling with random up to theta_max.")
        # Если не удалось собрать достаточно, заполняем оставшиеся случайными значениями в пределах theta_max_rad
        # Это не идеально, но лучше, чем ничего или зависание.
        remaining = num_samples - count
        samples[count:] = np.random.uniform(0, theta_max_rad, remaining)
    return samples

def sample_uniform_solid_angle_theta_vectorized(theta_max_rad: float, num_samples: int) -> np.ndarray:
    """
    Samples emission angles theta for uniform distribution over a solid angle cone (vectorized).
    """
    if math.isclose(theta_max_rad, 0.0):
        return np.zeros(num_samples)
        
    cos_theta_max = math.cos(theta_max_rad)
    denominator = 1.0 - cos_theta_max
    if math.isclose(denominator, 0.0):
        return np.zeros(num_samples)

    u = np.random.rand(num_samples)
    cos_theta = 1.0 - u * denominator
    cos_theta = np.clip(cos_theta, -1.0, 1.0) # Обеспечиваем валидный диапазон для acos
    return np.arccos(cos_theta)

def sample_emission_vector_vectorized(dist_type: str, theta_max_deg: float, params: dict, num_samples: int) -> np.ndarray:
    """
    Generates an array of random emission direction vectors (dx, dy, dz).
    Shape of returned array: (3, num_samples)
    """
    theta_max_rad = math.radians(theta_max_deg)
    
    if math.isclose(theta_max_rad, 0.0):
        theta_array = np.zeros(num_samples)
    else:
        if dist_type == config.DIST_GAUSSIAN:
            sigma_rad = math.radians(params.get('sigma', 1.0))
            theta_array = sample_gaussian_theta_vectorized(theta_max_rad, sigma_rad, num_samples)
        elif dist_type == config.DIST_COSINE_POWER:
            m = params.get('m_exp', 1.0)
            theta_array = sample_cosine_power_theta_vectorized(theta_max_rad, m, num_samples)
        elif dist_type == config.DIST_UNIFORM_SOLID:
            theta_array = sample_uniform_solid_angle_theta_vectorized(theta_max_rad, num_samples)
        else:
            # print(f"Warning: Unknown distribution type '{dist_type}'. Falling back to Uniform Solid Angle.")
            theta_array = sample_uniform_solid_angle_theta_vectorized(theta_max_rad, num_samples)

    phi_array = np.random.uniform(0, 2 * math.pi, num_samples)

    sin_theta_array = np.sin(theta_array)
    cos_theta_array = np.cos(theta_array)
    
    dx_array = sin_theta_array * np.cos(phi_array)
    dy_array = sin_theta_array * np.sin(phi_array)
    dz_array = -cos_theta_array 

    return np.vstack((dx_array, dy_array, dz_array)) # Shape (3, num_samples)

def sample_source_position_vectorized(src_type: str, params: dict, num_samples: int) -> np.ndarray:
    """
    Generates an array of random starting positions on the source surface.
    Shape of returned array: (3, num_samples)
    """
    if src_type == config.SOURCE_POINT:
        return np.zeros((3, num_samples))

    elif src_type == config.SOURCE_RING:
        src_radius = params.get('src_diameter', 0.0) / 2.0
        if src_radius <= 0: return np.zeros((3, num_samples))
        phi_array = np.random.uniform(0, 2 * math.pi, num_samples)
        x_array = src_radius * np.cos(phi_array)
        y_array = src_radius * np.sin(phi_array)
        z_array = np.zeros(num_samples)
        return np.vstack((x_array, y_array, z_array))

    elif src_type == config.SOURCE_CIRCULAR:
        src_radius = params.get('src_diameter', 0.0) / 2.0
        if src_radius <= 0: return np.zeros((3, num_samples))
        # Для равномерного распределения по площади диска, r нужно генерировать из sqrt(uniform)
        r_squared_array = np.random.uniform(0, src_radius**2, num_samples)
        r_array = np.sqrt(r_squared_array)
        phi_array = np.random.uniform(0, 2 * math.pi, num_samples)
        x_array = r_array * np.cos(phi_array)
        y_array = r_array * np.sin(phi_array)
        z_array = np.zeros(num_samples)
        return np.vstack((x_array, y_array, z_array))

    elif src_type == config.SOURCE_LINEAR:
        src_length = params.get('src_length', 0.0)
        src_angle_deg = params.get('src_angle', 0.0)
        if src_length <= 0: return np.zeros((3, num_samples))

        t_array = np.random.uniform(-src_length / 2.0, src_length / 2.0, num_samples)
        local_pos_array = np.vstack((t_array, np.zeros(num_samples), np.zeros(num_samples))) # Shape (3, num_samples)

        if src_angle_deg != 0.0:
            rot_angle_rad = math.radians(src_angle_deg)
            rot_mat = rotation_matrix([0, 0, 1], rot_angle_rad) # (3,3)
            return rot_mat @ local_pos_array # (3,3) @ (3,N) -> (3,N)
        else:
            return local_pos_array
    else:
        # print(f"Warning: Unknown source type '{src_type}'. Treating as point source.")
        return np.zeros((3, num_samples))

# Старые одночастичные функции (можно оставить для справки или удалить, если не используются)
def sample_gaussian_theta(theta_max_rad: float, sigma_rad: float) -> float:
    if math.isclose(theta_max_rad, 0.0): return 0.0
    if sigma_rad <= 0: return 0.0
    while True:
        theta = abs(np.random.normal(0, sigma_rad))
        if theta <= theta_max_rad: return theta

def sample_cosine_power_theta(theta_max_rad: float, m: float) -> float:
    if math.isclose(theta_max_rad, 0.0): return 0.0
    if m < 0: m = 0
    while True:
        theta = np.random.uniform(0, theta_max_rad)
        try: prob = (math.cos(theta))**m
        except ValueError: prob = 0.0
        if np.random.rand() <= prob: return theta

def sample_uniform_solid_angle_theta(theta_max_rad: float) -> float:
    if math.isclose(theta_max_rad, 0.0): return 0.0
    cos_theta_max = math.cos(theta_max_rad)
    denominator = 1.0 - cos_theta_max
    if math.isclose(denominator, 0.0): return 0.0
    u = np.random.rand()
    cos_theta = 1.0 - u * denominator
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.acos(cos_theta)

def sample_emission_vector(dist_type: str, theta_max_deg: float, params: dict) -> tuple[float, float, float]:
    # Эта функция теперь может быть оберткой над векторизованной для одной частицы,
    # или оставаться для старого кода, если он где-то используется.
    # Для новой логики симуляции она не нужна.
    # print("Warning: sample_emission_vector (single) called. Use vectorized version for performance.")
    res_array = sample_emission_vector_vectorized(dist_type, theta_max_deg, params, 1)
    return res_array[0,0], res_array[1,0], res_array[2,0]

def sample_source_position(src_type: str, params: dict) -> np.ndarray:
    # Аналогично, эта функция может быть оберткой.
    # print("Warning: sample_source_position (single) called. Use vectorized version for performance.")
    res_array = sample_source_position_vectorized(src_type, params, 1)
    return res_array[:,0] # Возвращает 1D массив (3,)
