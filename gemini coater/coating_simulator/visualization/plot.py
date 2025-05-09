# coding: utf-8
# coating_simulator_project/coating_simulator/visualization/plot.py
"""
Functions for plotting simulation results using Matplotlib.
Includes profile smoothing based on provided parameters.
Designed to draw on a provided Figure object. Uniformity stats are calculated outside.
Version 12.14 - Corrected Y_mesh/X_mesh for linear target masking.
"""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import Normalize, LogNorm
import numpy as np
import math
import matplotlib.patches as patches

SCIPY_AVAILABLE_FLAG = False
try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE_FLAG = True
except ImportError:
    pass

try:
    from .. import config
except ImportError:
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from coating_simulator import config

def _smooth_profile(coords: np.ndarray, profile_data: np.ndarray,
                    method: str, smoothing_params: dict) -> np.ndarray:
    if method == "Без сглаживания" or profile_data is None or coords is None or len(profile_data) != len(coords):
        return profile_data
    valid_indices = np.isfinite(profile_data)
    if np.sum(valid_indices) < 3: # Need at least 3 points to smooth
        return profile_data
    coords_valid = coords[valid_indices]
    profile_valid = profile_data[valid_indices]
    smoothed_profile_output = profile_data.copy() # Start with a copy

    if method == "Savitzky-Golay":
        if SCIPY_AVAILABLE_FLAG:
            window = smoothing_params.get('window_length', 11)
            poly = smoothing_params.get('polyorder', 3)

            # Basic validation for SavGol params
            if window < 3 or window % 2 == 0: window = 5 # Must be odd and >=3
            if poly >= window: poly = window - 1
            
            num_valid_points = len(profile_valid)
            if window > num_valid_points: # Window cannot be larger than data
                window = num_valid_points if num_valid_points % 2 != 0 else num_valid_points -1
                if window < 3 : window = 3 # Ensure window is at least 3
            
            # Ensure window is odd after adjustment
            if window % 2 == 0 : 
                window = window -1 if window > 3 else 3

            if poly >= window and window > 0: poly = window -1 
            if poly < 0: poly = 0


            try:
                if window > 0 and poly >=0 and num_valid_points > window : # Ensure enough points for the window
                    smoothed_values = savgol_filter(profile_valid, window, poly)
                    smoothed_profile_output[valid_indices] = smoothed_values
                else:
                    # Not enough points for smoothing with current window, return raw valid part
                    # This case should ideally be handled by the checks above, but as a fallback:
                    print(f"Warning (plot.py): Not enough points ({num_valid_points}) for SavGol window ({window}). Returning raw profile.")
                    return profile_data # Or smoothed_profile_output which is a copy of raw
            except Exception as e_sg:
                print(f"Error during Savitzky-Golay smoothing: {e_sg}. Returning raw profile.")
                return profile_data
        else:
            # print("Warning (plot.py): SciPy not available for Savitzky-Golay. Returning raw profile.")
            return profile_data # SciPy not available

    elif method == "Полином. аппрокс.":
        degree = smoothing_params.get('degree', 5)
        if degree < 1: degree = 3 # Min degree 1

        num_valid_points = len(coords_valid)
        if num_valid_points <= degree: # Degree must be less than number of points
            degree = num_valid_points - 1 if num_valid_points > 1 else 1
        
        if degree <= 0: # If still not possible
            print(f"Warning (plot.py): Not enough points ({num_valid_points}) for polynomial degree ({degree}). Returning raw profile.")
            return profile_data

        try:
            coeffs = np.polyfit(coords_valid, profile_valid, degree)
            poly_func = np.poly1d(coeffs)
            # Apply polynomial to original coords (including NaNs, then re-mask)
            smoothed_values_poly = poly_func(coords) # This will interpolate through NaNs if coords are complete
            smoothed_values_poly[np.isnan(profile_data)] = np.nan # Re-apply NaNs
            return smoothed_values_poly
        except Exception as e_poly:
            print(f"Error during polynomial smoothing: {e_poly}. Returning raw profile.")
            return profile_data
    else: # "Без сглаживания" or unknown method
        return profile_data # Return the original (or its copy)
    
    return smoothed_profile_output


def plot_simulation_results(fig: Figure,
                            coverage_map: np.ndarray, # Карта покрытия (N_cells_y, N_cells_x)
                            x_coords: np.ndarray,     # Границы ячеек по X (N_edges_x)
                            y_coords: np.ndarray,     # Границы ячеек по Y (N_edges_y)
                            radius_grid: np.ndarray,  # Радиусы для ЦЕНТРОВ ячеек (N_cells_y, N_cells_x)
                            target_params: dict,
                            vis_params: dict,
                            profile_1d_coords: np.ndarray = None, # Координаты X для профиля (центры ячеек)
                            profile_1d_values: np.ndarray = None, # Значения Y для профиля
                            show_colorbar: bool = True,
                            plot_type: str = "both"):
    fig.clear()
    ax_map = None
    ax_profile = None

    data = coverage_map.astype(float) # (N_cells_y, N_cells_x)
    use_percent = vis_params.get('percent', config.VIS_DEFAULT_PERCENT)
    data_on_substrate_raw = data.copy() # This will be masked
    target_type_for_norm = target_params.get('target_type', config.TARGET_DISK)

    # --- Логика маскирования data_on_substrate_raw ---
    # coverage_map (и data) имеет размер (num_cells_y, num_cells_x)
    # radius_grid (переданный как radius_grid_centers) также (num_cells_y, num_cells_x)
    # x_coords, y_coords - это ГРАНИЦЫ ячеек, т.е. len(x_coords) = num_cells_x + 1

    if target_type_for_norm in [config.TARGET_DISK, config.TARGET_DOME]:
        diameter = target_params.get('diameter', 0.0)
        actual_profile_extent_radius = diameter / 2.0
        if actual_profile_extent_radius > 0 and radius_grid is not None:
            if radius_grid.shape == data_on_substrate_raw.shape:
                 data_on_substrate_raw[radius_grid > actual_profile_extent_radius] = np.nan
            else:
                print(f"Предупреждение (plot.py): Несовпадение форм radius_grid ({radius_grid.shape}) и data ({data_on_substrate_raw.shape}) для диска/купола.")
        elif actual_profile_extent_radius <= 0: # Если диаметр 0, вся подложка NaN
            data_on_substrate_raw[:] = np.nan

    elif target_type_for_norm == config.TARGET_LINEAR:
        length = target_params.get('length', 0.0)
        width = target_params.get('width', 0.0)
        if length > 0 and width > 0:
            # x_coords и y_coords - это границы. Нам нужны центры для meshgrid.
            if x_coords is not None and len(x_coords) > 1 and y_coords is not None and len(y_coords) > 1:
                x_centers = (x_coords[:-1] + x_coords[1:]) / 2.0 # Длина: num_cells_x
                y_centers = (y_coords[:-1] + y_coords[1:]) / 2.0 # Длина: num_cells_y
                
                # Meshgrid для центров. Y_mesh_centers.shape = (len(y_centers), len(x_centers))
                # т.е. (num_cells_y, num_cells_x), что совпадает с data_on_substrate_raw.shape
                Y_mesh_centers, X_mesh_centers = np.meshgrid(y_centers, x_centers, indexing='ij') 
                
                if Y_mesh_centers.shape == data_on_substrate_raw.shape:
                    data_on_substrate_raw[np.abs(Y_mesh_centers) > (width / 2.0)] = np.nan
                    data_on_substrate_raw[np.abs(X_mesh_centers) > (length / 2.0)] = np.nan
                else:
                    # Эта ошибка не должна возникать, если логика выше верна
                    print(f"Критическая ошибка (plot.py): Несовпадение форм Y/X_mesh_centers ({Y_mesh_centers.shape}) и data ({data_on_substrate_raw.shape}) для линейной мишени после расчета центров.")
            else: 
                print("Предупреждение (plot.py): Некорректные x_coords или y_coords для линейной мишени.")
                data_on_substrate_raw[:] = np.nan
        else: 
            data_on_substrate_raw[:] = np.nan

    elif target_type_for_norm == config.TARGET_PLANETARY:
        orbit_rad = target_params.get('orbit_diameter', 0.0) / 2.0
        planet_rad = target_params.get('planet_diameter', 0.0) / 2.0
        actual_profile_extent_radius = orbit_rad + planet_rad
        if actual_profile_extent_radius > 0 and radius_grid is not None:
            if radius_grid.shape == data_on_substrate_raw.shape:
                data_on_substrate_raw[radius_grid > actual_profile_extent_radius] = np.nan
            else:
                print(f"Предупреждение (plot.py): Несовпадение форм radius_grid ({radius_grid.shape}) и data ({data_on_substrate_raw.shape}) для планетарной мишени.")
        elif actual_profile_extent_radius <= 0:
            data_on_substrate_raw[:] = np.nan
    # --- Конец логики маскирования ---

    max_val_on_substrate_raw = np.nanmax(data_on_substrate_raw) if np.any(np.isfinite(data_on_substrate_raw)) else 0.0
    if max_val_on_substrate_raw <= 0: max_val_on_substrate_raw = 1.0

    if use_percent:
        display_data_map = data / max_val_on_substrate_raw * 100.0
        coverage_label = 'Покрытие (%)'
    else:
        display_data_map = data
        coverage_label = 'Количество частиц'

    if plot_type == "both":
        ax_map, ax_profile = fig.subplots(1, 2, gridspec_kw={'wspace': 0.3})
    elif plot_type == "map_only":
        ax_map = fig.add_subplot(111)
    elif plot_type == "profile_only":
        ax_profile = fig.add_subplot(111)
    else:
        raise ValueError(f"Некорректный plot_type: {plot_type}")

    if ax_map is not None:
        # x_coords и y_coords здесь - это ГРАНИЦЫ ячеек
        x_min_extent, x_max_extent = x_coords[0], x_coords[-1]
        y_min_extent, y_max_extent = y_coords[0], y_coords[-1]
        extent = [x_min_extent, x_max_extent, y_min_extent, y_max_extent]
        
        aspect_ratio_val = 'equal'
        use_logscale = vis_params.get('logscale', config.VIS_DEFAULT_LOGSCALE)
        vmin_log = None
        if use_logscale and np.any(display_data_map > 0):
            min_positive = np.nanmin(display_data_map[display_data_map > 0])
            vmin_log = min_positive * 0.1 if min_positive > 0 else 1e-3
            if vmin_log is not None and np.nanmax(display_data_map) is not None and vmin_log >= np.nanmax(display_data_map):
                vmin_log = np.nanmax(display_data_map) * 0.01
        norm = LogNorm(vmin=vmin_log, vmax=np.nanmax(display_data_map) if np.any(np.isfinite(display_data_map)) else None) if use_logscale and vmin_log is not None and np.nanmax(display_data_map) is not None and vmin_log < np.nanmax(display_data_map) else Normalize()
        
        im = ax_map.imshow(display_data_map, extent=extent, origin='lower', cmap='hot', norm=norm, aspect=aspect_ratio_val)
        
        if show_colorbar:
            cbar = fig.colorbar(im, ax=ax_map, label=coverage_label, shrink=0.7, aspect=18, pad=0.04)
            cbar.ax.tick_params(labelsize=7)
            cbar.set_label(coverage_label, size=8)
        ax_map.set_title('Карта покрытия', fontsize=9)
        ax_map.set_xlabel('X (мм)', fontsize=8)
        ax_map.set_ylabel('Y (мм)', fontsize=8)
        ax_map.tick_params(axis='both', which='major', labelsize=7)
        ax_map.grid(True, linestyle=':')
        
        outline_color = 'white'; outline_style = '--'; outline_width = 0.8
        target_type_for_outline = target_params.get('target_type', config.TARGET_DISK)
        if target_type_for_outline in [config.TARGET_DISK, config.TARGET_DOME]:
            diameter = target_params.get('diameter', 0.0); radius = diameter / 2.0
            if radius > 0: ax_map.add_patch(patches.Circle((0, 0), radius, ec=outline_color, fc='none', ls=outline_style, lw=outline_width))
        elif target_type_for_outline == config.TARGET_LINEAR:
            length = target_params.get('length', 0.0); width_param = target_params.get('width', 0.0)
            if length > 0 and width_param > 0: ax_map.add_patch(patches.Rectangle((-length/2, -width_param/2), length, width_param, ec=outline_color, fc='none', ls=outline_style, lw=outline_width))
        elif target_type_for_outline == config.TARGET_PLANETARY:
            orbit_diameter = target_params.get('orbit_diameter', 0.0)
            if orbit_diameter > 0: ax_map.add_patch(patches.Circle((0,0), orbit_diameter/2.0, ec=outline_color, fc='none', ls=':', lw=outline_width))

    if ax_profile is not None and profile_1d_coords is not None and profile_1d_values is not None:
        ax_profile.plot(profile_1d_coords, profile_1d_values, '-', color='orange', linewidth=1.2, label='Профиль')
        profile_x_label = "Позиция X (мм)"
        target_type_for_profile = target_params.get('target_type', config.TARGET_DISK)
        if target_type_for_profile == config.TARGET_LINEAR: profile_x_label = "Позиция X (мм)"
        ax_profile.set_title('Профиль покрытия', fontsize=9)
        ax_profile.set_xlabel(profile_x_label, fontsize=8)
        ax_profile.set_ylabel(coverage_label, fontsize=8)
        ax_profile.grid(True, linestyle=':')
        ax_profile.tick_params(axis='both', which='major', labelsize=7)
        if np.any(np.isfinite(profile_1d_values)):
            y_min_plot = np.nanmin(profile_1d_values); y_max_plot = np.nanmax(profile_1d_values)
            padding = (y_max_plot - y_min_plot) * 0.05 if (y_max_plot - y_min_plot) > 1e-6 else 0.1
            final_y_min = y_min_plot - padding; final_y_max = y_max_plot + padding
            if use_percent:
                final_y_min = max(0, final_y_min)
                final_y_max = min(110, final_y_max) if final_y_max > 0 else 10
                if final_y_max <= final_y_min : final_y_max = final_y_min + 10
            ax_profile.set_ylim(bottom=final_y_min, top=final_y_max)
        else:
            ax_profile.set_ylim(0, 1 if not use_percent else 10)
        if ax_profile.has_data(): ax_profile.legend(fontsize='xx-small')
    elif ax_profile is not None :
        ax_profile.text(0.5, 0.5, "Данные профиля отсутствуют", ha='center', va='center', transform=ax_profile.transAxes, fontsize=8)
        ax_profile.set_title('Профиль покрытия', fontsize=9)
        ax_profile.set_xlabel("Позиция (мм)", fontsize=8)
        ax_profile.set_ylabel(coverage_label, fontsize=8)
        ax_profile.grid(True, linestyle=':')
        ax_profile.tick_params(axis='both', which='major', labelsize=7)

def calculate_uniformity_stats(profile_values: np.ndarray) -> dict:
    stats = {};
    if profile_values is None or len(profile_values) < 2: return stats
    valid_profile = profile_values[np.isfinite(profile_values)]
    if valid_profile.size >= 2:
        t_max = np.max(valid_profile); t_min = np.min(valid_profile)
        t_mean = np.mean(valid_profile); t_std = np.std(valid_profile)
        if (t_max + t_min) != 0: stats['U1'] = (t_max - t_min) / (t_max + t_min) * 100.0
        else: stats['U1'] = np.nan
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
    if result_val is not None and not np.isnan(result_val):
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

