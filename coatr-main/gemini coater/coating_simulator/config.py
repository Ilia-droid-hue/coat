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
    "src_x": 0.0, # Изменено на 0 для симметричного старта линейного движения по умолчанию
    "src_y": 0.0,
    "src_z": 900.0,
    "rot_x": 0.0,
    "rot_y": 0.0,
    "src_diameter": 100.0,
    "cone_angle": 60.0,
    "focus_point": "∞",
    "src_length": 100.0,
    "src_angle": 0.0,
}

DEFAULT_EMISSION_PARAMS = {
    "max_theta": 30.0, 
    "particles": 10000, # Будет преобразовано в 1 x 10^4 в GUI
    "sigma": 1.0,
    "m_exp": 1.0,
}

DEFAULT_PROCESSING_PARAMS = {
    "rpm": 10.0,
    "rpm_disk": 10.0,
    "rpm_orbit": 1.0,
    "speed": 100.0, # Увеличена скорость по умолчанию для линейного
    "time": 2.0,
    "mini_batch_size": 64, # <--- НОВЫЙ ПАРАМЕТР: Размер мини-пачки для пересчета трансформаций
}

# --- Simulation Parameters ---
SIM_GRID_SIZE = 200
# SIM_TRACE_STEPS = 500 # Больше не используется с аналитическим методом
# SIM_TRACE_MAX_DIST = 2000.0 # Больше не используется с аналитическим методом
SIM_INTERSECTION_TOLERANCE = 1e-3 # Уменьшена для большей точности с аналитикой
SIM_PROGRESS_INTERVAL_PERCENT = 5 # Для _run_simulation_loop (старый), в MP обновляется чаще

# --- Visualization Parameters ---
VIS_PROFILE_BINS = 100
VIS_DEFAULT_PERCENT = True
VIS_DEFAULT_LOGSCALE = False
VIS_DEFAULT_SHOW3D = False

# --- Special Values ---
INFINITY_SYMBOL = "∞"

# --- Helper Functions (без изменений) ---
def calculate_focus_point(diameter, cone_angle_deg):
    if cone_angle_deg <= 0: return INFINITY_SYMBOL
    try:
        phi_rad = math.radians(cone_angle_deg)
        tan_half_phi = math.tan(phi_rad / 2.0)
        if abs(tan_half_phi) < 1e-9: return INFINITY_SYMBOL
        L = (diameter / 2.0) / tan_half_phi
        return f"{L:.2f}"
    except (ValueError, ZeroDivisionError): return INFINITY_SYMBOL

def calculate_cone_angle(diameter, focus_point_L):
    try:
        L_float = float(focus_point_L)
        if L_float <= 1e-9: return "N/A"
        if diameter <= 0: return "N/A"
        atan_arg = (diameter / 2.0) / L_float
        if abs(atan_arg) > 1e9: return "N/A"
        phi_rad = 2 * math.atan(atan_arg)
        return f"{math.degrees(phi_rad):.2f}"
    except ValueError:
         if focus_point_L == INFINITY_SYMBOL: return "0.00"
         return "N/A"
    except ZeroDivisionError:
         return "N/A"
