# coating_simulator_project/coating_simulator/core/distribution.py
"""
Functions for generating particle distributions and performing rotations.
Функции для генерации распределений частиц и выполнения поворотов.
"""

import numpy as np
import math
# Используем относительный импорт для доступа к config внутри пакета
from .. import config

def rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """
    Calculates the rotation matrix for rotating around a given axis by an angle.
    Вычисляет матрицу поворота вокруг заданной оси на заданный угол.

    Args:
        axis: The axis of rotation (3D vector). Ось вращения (3D вектор).
        angle_rad: The angle of rotation in radians. Угол поворота в радианах.

    Returns:
        The 3x3 rotation matrix. Матрица поворота 3x3.
    """
    axis = np.asarray(axis, dtype=float)
    # Normalize the axis vector (important!)
    # Нормализуем вектор оси (важно!)
    norm = np.linalg.norm(axis)
    if norm == 0:
        # Return identity matrix if axis is zero vector
        # Возвращаем единичную матрицу, если ось - нулевой вектор
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

def sample_gaussian_theta(theta_max_rad: float, sigma_rad: float) -> float:
    """
    Samples an emission angle theta from a truncated Gaussian distribution.
    Генерирует случайный угол эмиссии theta из усеченного Гауссова распределения.
    Uses rejection sampling. Может быть неэффективным для больших theta_max / sigma.

    Args:
        theta_max_rad: Maximum allowed angle in radians. Максимальный угол в радианах.
        sigma_rad: Standard deviation of the Gaussian distribution in radians. Стандартное отклонение Гаусса в радианах.

    Returns:
        A sampled angle theta in radians (0 <= theta <= theta_max_rad).
        Случайный угол theta в радианах (0 <= theta <= theta_max_rad).
    """
    # Handle collimated beam case
    # Обработка случая коллимированного пучка
    if math.isclose(theta_max_rad, 0.0):
        return 0.0
        
    if sigma_rad <= 0:
        # If sigma is zero or negative, but theta_max is not zero, 
        # it implies a perfectly collimated beam within a non-zero cone, which is ambiguous.
        # Return 0 for simplicity, representing the central axis.
        # Если sigma ноль или отрицательна, но theta_max не ноль,
        # это подразумевает идеально коллимированный пучок внутри ненулевого конуса, что неоднозначно.
        # Возвращаем 0 для простоты, представляя центральную ось.
        print("Warning: sigma_rad <= 0 for Gaussian distribution. Returning theta=0.")
        return 0.0 

    while True:
        # Sample from a normal distribution (absolute value for angle)
        theta = abs(np.random.normal(0, sigma_rad))
        # Reject samples outside the max angle
        if theta <= theta_max_rad:
            return theta

def sample_cosine_power_theta(theta_max_rad: float, m: float) -> float:
    """
    Samples an emission angle theta according to a cosine-power distribution (cos(theta)^m).
    Генерирует случайный угол эмиссии theta согласно степенному косинусному распределению (cos(theta)^m).
    Uses rejection sampling. Может быть неэффективным для больших m или theta_max_rad близких к pi/2.

    Args:
        theta_max_rad: Maximum allowed angle in radians. Максимальный угол в радианах.
        m: The exponent in the cosine-power distribution. Экспонента в степенном косинусном распределении.

    Returns:
        A sampled angle theta in radians (0 <= theta <= theta_max_rad).
        Случайный угол theta в радианах (0 <= theta <= theta_max_rad).
    """
    # Handle collimated beam case
    # Обработка случая коллимированного пучка
    if math.isclose(theta_max_rad, 0.0):
        return 0.0

    if m < 0:
        print("Warning: m < 0 for Cosine-power distribution. Using m=0.")
        m = 0
        
    # Max probability is 1 (at theta=0)
    while True:
        # Uniformly sample theta within the allowed range
        theta = np.random.uniform(0, theta_max_rad)
        # Calculate the probability density (relative to max=1)
        try:
            prob = (math.cos(theta))**m
        except ValueError: # Handle potential domain error if theta slightly > pi/2 due to float issues
             prob = 0.0
        # Accept or reject based on a uniform random number
        if np.random.rand() <= prob:
            return theta

def sample_uniform_solid_angle_theta(theta_max_rad: float) -> float:
    """
    Samples an emission angle theta corresponding to a uniform distribution over a solid angle cone.
    Генерирует случайный угол эмиссии theta, соответствующий равномерному распределению по телесному углу конуса.
    dN ~ sin(theta) dtheta dphi. Интегрируя по phi и нормализуя, получаем CDF ~ 1 - cos(theta).

    Args:
        theta_max_rad: Maximum angle of the cone in radians. Максимальный угол конуса в радианах.

    Returns:
        A sampled angle theta in radians (0 <= theta <= theta_max_rad).
        Случайный угол theta в радианах (0 <= theta <= theta_max_rad).
    """
    # Handle collimated beam case
    # Обработка случая коллимированного пучка
    if math.isclose(theta_max_rad, 0.0):
        return 0.0
        
    cos_theta_max = math.cos(theta_max_rad)
    # Avoid division by zero if theta_max is pi (cos_theta_max = -1) -> 1 - cos_theta_max = 2
    # Avoid issues if theta_max is ~0 (cos_theta_max = 1) -> 1 - cos_theta_max = 0
    denominator = 1.0 - cos_theta_max
    if math.isclose(denominator, 0.0): # Happens if theta_max_rad is very close to 0
        return 0.0 # Collimated beam

    u = np.random.rand() # Uniform random number between 0 and 1
    cos_theta = 1.0 - u * denominator
    # Ensure the argument for acos is within the valid range [-1, 1]
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.acos(cos_theta)


def sample_emission_vector(dist_type: str, theta_max_deg: float, params: dict) -> tuple[float, float, float]:
    """
    Generates a random emission direction vector (dx, dy, dz) based on the distribution type.
    Генерирует случайный вектор направления эмиссии (dx, dy, dz) на основе типа распределения.
    The vector points generally towards the negative Z direction.
    Вектор в основном направлен в отрицательном направлении оси Z.

    Args:
        dist_type: Type of angular distribution (from config). Тип углового распределения (из config).
        theta_max_deg: Maximum emission angle (half-angle) in degrees. Максимальный угол эмиссии (половинный) в градусах.
        params: Dictionary containing distribution-specific parameters like 'sigma' or 'm_exp'.
                Словарь, содержащий специфичные для распределения параметры, такие как 'sigma' или 'm_exp'.

    Returns:
        A tuple (dx, dy, dz) representing the unit direction vector.
        Кортеж (dx, dy, dz), представляющий единичный вектор направления.
    """
    theta_max_rad = math.radians(theta_max_deg)
    
    # Handle collimated beam case directly
    # Обрабатываем случай коллимированного пучка напрямую
    if math.isclose(theta_max_rad, 0.0):
        theta = 0.0
    else:
        # Sample theta based on distribution type
        if dist_type == config.DIST_GAUSSIAN:
            sigma_rad = math.radians(params.get('sigma', 1.0)) 
            theta = sample_gaussian_theta(theta_max_rad, sigma_rad)
        elif dist_type == config.DIST_COSINE_POWER:
            m = params.get('m_exp', 1.0) 
            theta = sample_cosine_power_theta(theta_max_rad, m)
        elif dist_type == config.DIST_UNIFORM_SOLID:
            theta = sample_uniform_solid_angle_theta(theta_max_rad)
        else:
            print(f"Warning: Unknown distribution type '{dist_type}'. Falling back to Uniform Solid Angle.")
            theta = sample_uniform_solid_angle_theta(theta_max_rad)

    # Sample the azimuthal angle phi uniformly
    phi = np.random.uniform(0, 2 * math.pi)

    # Convert spherical coordinates (theta, phi) to Cartesian direction vector
    # Note: theta is angle from negative Z-axis
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta) # Calculate once
    
    dx = sin_theta * math.cos(phi)
    dy = sin_theta * math.sin(phi)
    dz = -cos_theta # Points towards negative Z (if theta=0, dz=-1)

    return dx, dy, dz


def sample_source_position(src_type: str, params: dict) -> np.ndarray:
    """
    Generates a random starting position on the source surface based on its type.
    Генерирует случайную начальную позицию на поверхности источника в зависимости от его типа.
    Position is relative to the source's local origin (before translation/rotation).
    Позиция относительна локальному началу координат источника (до смещения/поворота).

    Args:
        src_type: The type of the source (from config). Тип источника (из config).
        params: Dictionary containing source-specific parameters like 'src_diameter', 'src_length', 'src_angle'.
                Словарь, содержащий специфичные для источника параметры, такие как 'src_diameter', 'src_length', 'src_angle'.

    Returns:
        A numpy array [x, y, z] representing the starting position in local coordinates.
        Массив numpy [x, y, z], представляющий начальную позицию в локальных координатах.
    """
    if src_type == config.SOURCE_POINT:
        return np.zeros(3)

    elif src_type == config.SOURCE_RING:
        src_radius = params.get('src_diameter', 0.0) / 2.0
        if src_radius <= 0: return np.zeros(3) 
        phi = np.random.uniform(0, 2 * math.pi)
        return np.array([src_radius * math.cos(phi), src_radius * math.sin(phi), 0.0])

    elif src_type == config.SOURCE_CIRCULAR:
        src_radius = params.get('src_diameter', 0.0) / 2.0
        if src_radius <= 0: return np.zeros(3)
        r_squared = np.random.uniform(0, src_radius**2)
        r = math.sqrt(r_squared)
        phi = np.random.uniform(0, 2 * math.pi)
        return np.array([r * math.cos(phi), r * math.sin(phi), 0.0])

    elif src_type == config.SOURCE_LINEAR:
        src_length = params.get('src_length', 0.0)
        src_angle_deg = params.get('src_angle', 0.0)
        if src_length <= 0: return np.zeros(3)

        t = np.random.uniform(-src_length / 2.0, src_length / 2.0)
        local_pos = np.array([t, 0.0, 0.0])

        if src_angle_deg != 0.0:
            rot_angle_rad = math.radians(src_angle_deg)
            rot_mat = rotation_matrix([0, 0, 1], rot_angle_rad)
            return rot_mat.dot(local_pos)
        else:
            return local_pos
    else:
        print(f"Warning: Unknown source type '{src_type}'. Treating as point source.")
        return np.zeros(3)

