# coating_simulator_project/coating_simulator/config.py
"""
Configuration constants for the coating simulator.
Константы конфигурации для симулятора покрытия.
"""

import math

# --- Target Types ---
TARGET_DISK = "диск"
TARGET_DOME = "купол"
TARGET_LINEAR = "линейное перемещение"
TARGET_PLANETARY = "планетарный"

TARGET_TYPES = [TARGET_DISK, TARGET_DOME, TARGET_LINEAR, TARGET_PLANETARY]

# --- Source Types ---
SOURCE_POINT = "точечный"
SOURCE_RING = "кольцевой"
SOURCE_CIRCULAR = "круглый"
SOURCE_LINEAR = "линейный"

SOURCE_TYPES = [SOURCE_POINT, SOURCE_RING, SOURCE_CIRCULAR, SOURCE_LINEAR]

# --- Emission Distribution Types ---
DIST_GAUSSIAN = "Gaussian beam"
DIST_COSINE_POWER = "Cosine-power (cosᵐθ)"
DIST_UNIFORM_SOLID = "Uniform solid angle"

DISTRIBUTION_TYPES = [DIST_GAUSSIAN, DIST_COSINE_POWER, DIST_UNIFORM_SOLID]

# --- Default Values ---
DEFAULT_TARGET_PARAMS = {
    "diameter": 3200.0,
    "dome_radius": 2000.0,
    "length": 4000.0,
    "width": 3000.0,
    "orbit_diameter": 4000.0,
    "planet_disk_diameter": 3200.0,
}

DEFAULT_SOURCE_PARAMS = {
    "src_x": 0.0,
    "src_y": 0.0,
    "src_z": 900.0,
    "rot_x": 0.0,
    "rot_y": 0.0,
    "src_diameter": 100.0,
    "cone_angle": 60.0,
    "focus_point": "∞", # Используем константу INFINITY_SYMBOL ниже
    "src_length": 100.0,
    "src_angle": 0.0,
}

DEFAULT_EMISSION_PARAMS = {
    "max_theta": 30.0, # Половинный угол эмиссии в градусах
    "particles": 10000,
    "sigma": 1.0,      # Для Gaussian beam, в градусах
    "m_exp": 1.0,      # Для Cosine-power
}

DEFAULT_PROCESSING_PARAMS = {
    "rpm": 10.0,
    "rpm_disk": 10.0,       # Для планетарного
    "rpm_orbit": 1.0,       # Для планетарного
    "speed": 100.0,         # Для линейного перемещения (мм/с)
    "time": 2.0,            # Общее время симуляции (с)
    "mini_batch_size": 64,  # Размер мини-пакета для пересчета трансформаций
}

# --- Simulation Parameters ---
SIM_GRID_SIZE = 200 # Количество ГРАНИЦ ячеек сетки (количество ячеек будет на 1 меньше)
SIM_INTERSECTION_TOLERANCE = 1e-3 # Допуск для проверки пересечения луча с поверхностью
SIM_PROGRESS_INTERVAL_PERCENT = 5 # Не используется напрямую в MP, но может быть полезно

# --- Visualization Parameters ---
VIS_PROFILE_BINS = 100
VIS_DEFAULT_PERCENT = True
VIS_DEFAULT_LOGSCALE = False
VIS_DEFAULT_SHOW3D = False # Не используется в текущей 2D визуализации результатов

# НОВЫЕ КОНСТАНТЫ для окна результатов (для будущего использования)
PLOT_TYPES_AVAILABLE = ["2D Карта", "3D Поверхность", "Профиль"]
DEFAULT_PLOT_TYPE = PLOT_TYPES_AVAILABLE[0]

COLORMAPS_AVAILABLE = ["viridis", "plasma", "magma", "cividis", "gray", "jet"]
DEFAULT_COLORMAP = COLORMAPS_AVAILABLE[0]


# --- Special Values ---
INFINITY_SYMBOL = "∞"
# Обновляем значение по умолчанию, если оно использует этот символ
if DEFAULT_SOURCE_PARAMS["focus_point"] == "∞":
    DEFAULT_SOURCE_PARAMS["focus_point"] = INFINITY_SYMBOL


# --- Helper Functions (для SourceFrame, расчет фокуса и угла конуса) ---
def calculate_focus_point(diameter_str, cone_angle_deg_str):
    """
    Рассчитывает точку фокуса L по диаметру D и углу конуса φ.
    Возвращает строку для отображения в GUI.
    """
    try:
        D = float(diameter_str)
        phi_deg = float(cone_angle_deg_str)
        if D <= 0: return "N/A (D≤0)"
        if phi_deg <= 0 or phi_deg >= 180: return INFINITY_SYMBOL # Параллельные лучи
        
        phi_rad = math.radians(phi_deg)
        tan_half_phi = math.tan(phi_rad / 2.0)
        if abs(tan_half_phi) < 1e-9: return INFINITY_SYMBOL # Почти параллельные
        
        L = (D / 2.0) / tan_half_phi
        return f"{L:.2f}"
    except ValueError:
        return "N/A" # Ошибка ввода
    except ZeroDivisionError: # На всякий случай
        return INFINITY_SYMBOL

def calculate_cone_angle(diameter_str, focus_point_L_str):
    """
    Рассчитывает угол конуса φ по диаметру D и точке фокуса L.
    Возвращает строку для отображения в GUI.
    """
    try:
        D = float(diameter_str)
        if D <= 0: return "N/A (D≤0)"

        if focus_point_L_str == INFINITY_SYMBOL:
            return "0.00" # Параллельные лучи соответствуют углу конуса 0

        L = float(focus_point_L_str)
        if abs(L) < 1e-9 : return "N/A (L≈0)" # Фокус в центре источника

        atan_arg = (D / 2.0) / L
        half_phi_rad = math.atan(atan_arg)
        phi_deg = math.degrees(2 * half_phi_rad)

        if abs(phi_deg) > 180: return "N/A (L слишком мал)" # Угол слишком большой
        return f"{phi_deg:.2f}"

    except ValueError: # L_str не float и не INFINITY_SYMBOL
        return "N/A"
    except ZeroDivisionError: # D/L или L=0
        if abs(D) > 1e-9 : return "180.00" # Если диаметр есть, а фокус в центре, то угол 180
        else: return "0.00" # Если и диаметр 0, то угол 0

