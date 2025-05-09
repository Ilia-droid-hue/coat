# coding: utf-8
# coating_simulator_project/coating_simulator/visualization/plot.py
"""
Functions for plotting simulation results using Matplotlib.
Includes profile smoothing based on provided parameters.
Designed to draw on a provided Figure object. Uniformity stats are calculated outside.
"""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import Normalize, LogNorm
import numpy as np
import math
import matplotlib.patches as patches 

# Проверка доступности SciPy вынесена в ResultsWindow, здесь просто используем флаг
SCIPY_AVAILABLE_FLAG = False 
try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE_FLAG = True
except ImportError:
    pass 

try:
    from .. import config
except ImportError:
    # Fallback для возможности запуска этого файла отдельно для тестирования (если потребуется)
    # В реальном приложении этот fallback не должен срабатывать.
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir)) # Up two levels
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from coating_simulator import config


# --- Функция сглаживания профиля ---
def _smooth_profile(coords: np.ndarray, profile_data: np.ndarray,
                    method: str, smoothing_params: dict) -> np.ndarray:
    """
    Сглаживает 1D профиль на основе выбранного метода и параметров.

    Args:
        coords: 1D массив координат X профиля.
        profile_data: 1D массив "сырых" значений Y профиля (может содержать NaN).
        method: Строка, указывающая метод сглаживания ("Savitzky-Golay", "Полином. аппрокс.", "Без сглаживания").
        smoothing_params: Словарь с параметрами для метода (например, {'window_length': 11, 'polyorder': 3}).

    Returns:
        1D массив сглаженных значений Y профиля.
    """
    if method == "Без сглаживания" or profile_data is None or coords is None or len(profile_data) != len(coords):
        return profile_data

    valid_indices = np.isfinite(profile_data)
    if np.sum(valid_indices) < 3: 
        print("Предупреждение: Недостаточно валидных точек для сглаживания, возвращается сырой профиль.")
        return profile_data

    coords_valid = coords[valid_indices]
    profile_valid = profile_data[valid_indices]
    smoothed_profile_output = profile_data.copy() 

    if method == "Savitzky-Golay":
        if SCIPY_AVAILABLE_FLAG:
            window = smoothing_params.get('window_length', 11)
            poly = smoothing_params.get('polyorder', 3)
            
            if window < 3 or window % 2 == 0:
                print(f"Предупреждение: Некорректная длина окна SavGol ({window}), используется 5.")
                window = 5
            if poly >= window:
                poly = window - 1
                print(f"Предупреждение: Порядок полинома SavGol ({poly+1}) >= длины окна ({window}), используется {poly}.")
            
            # Проверка, чтобы окно не было больше данных и было нечетным
            num_valid_points = len(profile_valid)
            if window > num_valid_points:
                window = num_valid_points if num_valid_points % 2 != 0 else num_valid_points -1
                if window < 3 : window = 3 # Минимальное окно
                print(f"Предупреждение: Длина окна SavGol ({smoothing_params.get('window_length', 11)}) > кол-ва точек ({num_valid_points}), используется {window}.")
            if window % 2 == 0 : # Дополнительная проверка на нечетность
                window = window -1 if window > 3 else 3

            if poly >= window and window > 0: poly = window -1
            if poly < 0: poly = 0


            try:
                if window > 0 and poly >=0 and num_valid_points > window : 
                    smoothed_values = savgol_filter(profile_valid, window, poly)
                    smoothed_profile_output[valid_indices] = smoothed_values
                    return smoothed_profile_output
                else:
                    print(f"Предупреждение: Недостаточно точек ({num_valid_points}) для окна SavGol ({window}) или некорректный polyorder ({poly}). Возвращается сырой профиль.")
                    return profile_data
            except Exception as e:
                print(f"Ошибка при сглаживании Savitzky-Golay: {e}. Возвращается сырой профиль.")
                return profile_data
        else:
            print("Предупреждение: SciPy недоступен для Savitzky-Golay. Возвращается сырой профиль.")
            return profile_data

    elif method == "Полином. аппрокс.":
        degree = smoothing_params.get('degree', 5)
        if degree < 1:
            print(f"Предупреждение: Некорректная степень полинома ({degree}), используется 3.")
            degree = 3
        
        num_valid_points = len(coords_valid)
        if num_valid_points <= degree:
            new_degree = num_valid_points - 1 if num_valid_points > 1 else 1
            print(f"Предупреждение: Степень полинома ({degree}) >= кол-ва точек ({num_valid_points}). Используется {new_degree}.")
            degree = new_degree
        
        if degree <= 0: # Если после коррекции степень стала некорректной
            print("Предупреждение: Невозможно выполнить полиномиальную аппроксимацию с такой степенью. Возвращается сырой профиль.")
            return profile_data

        try:
            coeffs = np.polyfit(coords_valid, profile_valid, degree)
            poly_func = np.poly1d(coeffs)
            smoothed_values_poly = poly_func(coords) # Применяем ко всем исходным координатам
            smoothed_values_poly[np.isnan(profile_data)] = np.nan # Восстанавливаем NaN
            return smoothed_values_poly
        except Exception as e_poly:
            print(f"Ошибка при полиномиальном сглаживании: {e_poly}. Возвращается сырой профиль.")
            return profile_data
    else:
        print(f"Предупреждение: Неизвестный метод сглаживания '{method}'. Возвращается сырой профиль.")
        return profile_data

    return smoothed_profile_output


# --- Основная функция отрисовки (plot_simulation_results) ---
def plot_simulation_results(fig: Figure,
                            coverage_map: np.ndarray,
                            x_coords: np.ndarray,
                            y_coords: np.ndarray,
                            radius_grid: np.ndarray,
                            target_params: dict,
                            vis_params: dict,
                            profile_1d_coords: np.ndarray, # Координаты для 1D профиля
                            profile_1d_values: np.ndarray, # Значения для 1D профиля (уже обработанные/сглаженные)
                            show_colorbar: bool = True):   # Новый параметр для управления colorbar
    """
    Отображает 2D карту покрытия и 1D профиль на Figure.
    """
    fig.clear()
    # Создаем два subplots: один для карты, другой для профиля
    ax_map, ax_profile = fig.subplots(1, 2, gridspec_kw={'wspace': 0.3}) # ax_map = ax[0], ax_profile = ax[1]

    data = coverage_map.astype(float) 
    use_percent = vis_params.get('percent', config.VIS_DEFAULT_PERCENT)
    use_logscale = vis_params.get('logscale', config.VIS_DEFAULT_LOGSCALE)

    data_on_substrate_raw = data.copy() 
    target_type = target_params.get('target_type', config.TARGET_DISK)
    actual_profile_extent_radius = 0 
    
    if target_type in [config.TARGET_DISK, config.TARGET_DOME]:
        diameter = target_params.get('diameter', 0.0); actual_profile_extent_radius = diameter / 2.0
        if actual_profile_extent_radius > 0: data_on_substrate_raw[radius_grid > actual_profile_extent_radius] = np.nan
        else: data_on_substrate_raw[:] = np.nan
    elif target_type == config.TARGET_LINEAR:
        length = target_params.get('length', 0.0); width = target_params.get('width', 0.0)
        actual_profile_extent_radius = length / 2.0 # Для линейного профиля это будет длина
        if length > 0 and width > 0:
            # Для линейной мишени, x_coords и y_coords определяют сетку
            # radius_grid здесь может быть не так релевантен, как прямые координаты
            Y_mesh, X_mesh = np.meshgrid(y_coords, x_coords, indexing='ij') # Важно: indexing='ij' если coverage_map (rows, cols) ~ (y, x)
            data_on_substrate_raw[np.abs(Y_mesh) > (width / 2.0)] = np.nan
            data_on_substrate_raw[np.abs(X_mesh) > (length / 2.0)] = np.nan
        else: data_on_substrate_raw[:] = np.nan
    elif target_type == config.TARGET_PLANETARY:
        orbit_rad = target_params.get('orbit_diameter', 0.0) / 2.0; planet_rad = target_params.get('planet_diameter', 0.0) / 2.0
        actual_profile_extent_radius = orbit_rad + planet_rad
        if actual_profile_extent_radius > 0: data_on_substrate_raw[radius_grid > actual_profile_extent_radius] = np.nan
        else: data_on_substrate_raw[:] = np.nan

    max_val_on_substrate_raw = np.nanmax(data_on_substrate_raw) if np.any(np.isfinite(data_on_substrate_raw)) else 0.0
    if max_val_on_substrate_raw <= 0: max_val_on_substrate_raw = 1.0 # Избегаем деления на ноль

    if use_percent:
        display_data_map = data / max_val_on_substrate_raw * 100.0 
        coverage_label = 'Покрытие (%)'
    else:
        display_data_map = data
        coverage_label = 'Количество частиц'

    # --- Карта покрытия (на ax_map) ---
    x_min, x_max = x_coords[0], x_coords[-1]; y_min, y_max = y_coords[0], y_coords[-1]
    extent = [x_min, x_max, y_min, y_max]
    aspect_ratio = 'equal' if abs((x_max - x_min) - (y_max - y_min)) < 1e-6 else 'auto'
    vmin_log = None
    if use_logscale and np.any(display_data_map > 0):
        min_positive = np.nanmin(display_data_map[display_data_map > 0])
        vmin_log = min_positive * 0.1 if min_positive > 0 else 1e-3
    norm = LogNorm(vmin=vmin_log, vmax=np.nanmax(display_data_map) if np.any(np.isfinite(display_data_map)) else None) if use_logscale else Normalize()

    im = ax_map.imshow(display_data_map, extent=extent, origin='lower', cmap='hot', norm=norm, aspect=aspect_ratio)
    
    # --- Управление отображением Colorbar ---
    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax_map, label=coverage_label, shrink=0.7, aspect=15)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label(coverage_label, size=9)
    # -----------------------------------------

    ax_map.set_title('Карта покрытия', fontsize=10)
    ax_map.set_xlabel('X (мм)', fontsize=9)
    ax_map.set_ylabel('Y (мм)', fontsize=9)
    ax_map.tick_params(axis='both', which='major', labelsize=8)
    ax_map.grid(True, linestyle=':') # Добавим сетку на карту

    # --- Контур подложки (на ax_map) ---
    outline_color = 'white'; outline_style = '--'; outline_width = 1.0
    if target_type in [config.TARGET_DISK, config.TARGET_DOME]:
        diameter = target_params.get('diameter', 0.0); radius = diameter / 2.0
        if radius > 0: ax_map.add_patch(patches.Circle((0, 0), radius, ec=outline_color, fc='none', ls=outline_style, lw=outline_width))
    elif target_type == config.TARGET_LINEAR:
        length = target_params.get('length', 0.0); width = target_params.get('width', 0.0)
        if length > 0 and width > 0: ax_map.add_patch(patches.Rectangle((-length/2, -width/2), length, width, ec=outline_color, fc='none', ls=outline_style, lw=outline_width))
    elif target_type == config.TARGET_PLANETARY:
        orbit_diameter = target_params.get('orbit_diameter', 0.0)
        if orbit_diameter > 0: ax_map.add_patch(patches.Circle((0,0), orbit_diameter/2.0, ec=outline_color, fc='none', ls=':', lw=outline_width))


    # --- Отображение 1D профиля (на ax_profile) ---
    # profile_1d_coords и profile_1d_values передаются в функцию
    if profile_1d_coords is not None and profile_1d_values is not None:
        ax_profile.plot(profile_1d_coords, profile_1d_values, '-', color='orange', linewidth=1.5, label='Профиль')
        
        # Определение метки для оси X профиля
        profile_x_label = "Позиция X (мм)" # По умолчанию
        if target_type == config.TARGET_LINEAR:
            profile_x_label = "Позиция X (мм)"
        # Для дисковых и планетарных, если профиль радиальный, метка может быть "Радиус (мм)"
        # В текущей реализации results.py профиль всегда по X, так что "Позиция X (мм)" подходит.

        ax_profile.set_title('Профиль покрытия', fontsize=10)
        ax_profile.set_xlabel(profile_x_label, fontsize=9)
        ax_profile.set_ylabel(coverage_label, fontsize=9) 
        ax_profile.grid(True, linestyle=':')
        ax_profile.tick_params(axis='both', which='major', labelsize=8)
        
        if np.any(np.isfinite(profile_1d_values)):
            y_min_plot = np.nanmin(profile_1d_values)
            y_max_plot = np.nanmax(profile_1d_values)
            padding = (y_max_plot - y_min_plot) * 0.05 if (y_max_plot - y_min_plot) > 1e-6 else 0.1
            
            final_y_min = y_min_plot - padding
            final_y_max = y_max_plot + padding

            if use_percent: 
                final_y_min = max(0, final_y_min)
                final_y_max = min(110, final_y_max) if final_y_max > 0 else 10 
                if final_y_max <= final_y_min : final_y_max = final_y_min + 10 
            
            ax_profile.set_ylim(bottom=final_y_min, top=final_y_max)
        else: 
            ax_profile.set_ylim(0, 1 if not use_percent else 10)
        
        # Добавляем легенду, если есть что отображать
        if ax_profile.has_data():
            ax_profile.legend(fontsize='small')
    else:
        ax_profile.text(0.5, 0.5, "Данные профиля отсутствуют", ha='center', va='center', transform=ax_profile.transAxes)
        ax_profile.set_title('Профиль покрытия', fontsize=10)
        ax_profile.set_xlabel("Позиция (мм)", fontsize=9)
        ax_profile.set_ylabel(coverage_label, fontsize=9)
        ax_profile.grid(True, linestyle=':')
        ax_profile.tick_params(axis='both', which='major', labelsize=8)


def calculate_uniformity_stats(profile_values: np.ndarray) -> dict:
    stats = {};
    if profile_values is None or len(profile_values) < 2:
        return stats
    valid_profile = profile_values[np.isfinite(profile_values)]
    if valid_profile.size >= 2:
        t_max = np.max(valid_profile); t_min = np.min(valid_profile)
        t_mean = np.mean(valid_profile); t_std = np.std(valid_profile)
        if (t_max + t_min) != 0: stats['U1'] = (t_max - t_min) / (t_max + t_min) * 100.0
        else: stats['U1'] = np.nan # или 0, или другое значение по умолчанию
        if t_mean != 0: stats['U2'] = (t_max - t_min) / t_mean * 100.0
        else: stats['U2'] = np.nan
        if t_mean != 0: stats['U3'] = t_std / t_mean * 100.0
        else: stats['U3'] = np.nan
        if t_max != 0: stats['U4'] = t_min / t_max * 100.0
        else: stats['U4'] = np.nan
    return stats

def format_uniformity_stats(stats_dict: dict, selected_method: str) -> str:
    if not stats_dict: return "N/A (<2 т.)"
    result_val = stats_dict.get(selected_method)
    if result_val is not None and not np.isnan(result_val): # Проверка на NaN
        formula_text_map = {
            'U1': "$U_1=\\frac{Max-Min}{Max+Min}$", 'U2': "$U_2=\\frac{Max-Min}{\\bar{t}}$",
            'U3': "$U_3=\\frac{\\sigma}{\\bar{t}}$", 'U4': "$U_4=\\frac{Min}{Max}$"
        }
        formula_text = formula_text_map.get(selected_method, selected_method)
        return f"{formula_text} = {result_val:.1f} %"
    elif result_val is np.nan:
        return f"{selected_method}: N/A (деление на 0)"
    else:
        return f"{selected_method}: N/A (ошибка)"

