# coating_simulator_project/coating_simulator/visualization/export.py
"""
Function for exporting simulation results to CSV files.
Функция для экспорта результатов симуляции в файлы CSV.
"""

import numpy as np
from tkinter import filedialog
from datetime import datetime
import os # For joining paths

# Используем относительный импорт для доступа к config внутри пакета
from .. import config

def export_csv(coverage_map: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, radius_grid: np.ndarray,
                 target_params: dict, vis_params: dict):
    """
    Exports the coverage map and its radial/linear profile to CSV files.
    Экспортирует карту покрытия и ее радиальный/линейный профиль в файлы CSV.

    Args:
        coverage_map: 2D numpy array with particle counts per cell.
                      2D массив numpy со счетчиками частиц на ячейку.
        x_coords: 1D numpy array of X coordinates for the grid.
                  1D массив numpy координат X для сетки.
        y_coords: 1D numpy array of Y coordinates for the grid.
                  1D массив numpy координат Y для сетки.
        radius_grid: 2D numpy array with the radius for each cell center.
                     2D массив numpy с радиусом для центра каждой ячейки.
        target_params: Dictionary with target parameters (e.g., 'diameter', 'length', 'target_type').
                       Словарь с параметрами мишени (например, 'diameter', 'length', 'target_type').
        vis_params: Dictionary with visualization parameters ('percent').
                    Словарь с параметрами визуализации ('percent').
    """
    if coverage_map is None or x_coords is None or y_coords is None or radius_grid is None:
        print("Ошибка: Нет данных для экспорта.")
        # Optionally show a messagebox error here if called from GUI context
        # Опционально показать messagebox с ошибкой, если вызвано из контекста GUI
        # import tkinter.messagebox
        # tkinter.messagebox.showerror("Ошибка экспорта", "Нет данных для экспорта. Запустите симуляцию сначала.")
        return

    data = coverage_map.astype(float)
    use_percent = vis_params.get('percent', config.VIS_DEFAULT_PERCENT)

    # Normalize data if required
    # Нормализуем данные при необходимости
    max_coverage = np.max(data)
    if use_percent and max_coverage > 0:
        data_to_export = data / max_coverage * 100.0
        profile_data_source = data_to_export
        map_header = "Coverage Map (%)"
        profile_header = "radius_mm,coverage_percent" if target_params.get('target_type') != config.TARGET_LINEAR else "x_position_mm,coverage_percent"
    elif use_percent: # max_coverage is 0
        data_to_export = data # Export zeros
                               # Экспортируем нули
        profile_data_source = data_to_export
        map_header = "Coverage Map (%)"
        profile_header = "radius_mm,coverage_percent" if target_params.get('target_type') != config.TARGET_LINEAR else "x_position_mm,coverage_percent"
    else:
        data_to_export = coverage_map # Export raw counts
                                      # Экспортируем необработанные счетчики
        profile_data_source = data_to_export
        map_header = "Coverage Map (Particle Count)"
        profile_header = "radius_mm,particle_count" if target_params.get('target_type') != config.TARGET_LINEAR else "x_position_mm,particle_count"


    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Suggest a filename for the map
    # Предлагаем имя файла для карты
    map_filename_suggestion = f'coverage_map_{now}.csv'
    fname_map = filedialog.asksaveasfilename(
        defaultextension='.csv',
        initialfile=map_filename_suggestion,
        title="Сохранить карту покрытия как CSV",
        filetypes=[('CSV файлы', '*.csv'), ('Все файлы', '*.*')]
    )

    if fname_map:
        try:
            # Add header information (optional)
            # Добавляем информацию заголовка (опционально)
            header_lines = [
                f"# {map_header}",
                f"# Exported on: {now}",
                f"# Grid dimensions: {data_to_export.shape[0]} (Y) x {data_to_export.shape[1]} (X)",
                f"# X Coordinates range: {x_coords[0]:.2f} mm to {x_coords[-1]:.2f} mm",
                f"# Y Coordinates range: {y_coords[0]:.2f} mm to {y_coords[-1]:.2f} mm"
            ]
            np.savetxt(fname_map, data_to_export, delimiter=',', header='\n'.join(header_lines), comments='')
            print(f"Карта покрытия успешно сохранена в {fname_map}")
        except Exception as e:
            print(f"Ошибка при сохранении карты покрытия: {e}")
            # Consider showing error to user via messagebox if in GUI context
            # Рассмотреть возможность показа ошибки пользователю через messagebox, если в контексте GUI

    # --- Calculate and Export Profile ---
    bins = config.VIS_PROFILE_BINS
    profile = np.zeros(bins)

    target_type = target_params.get('target_type', config.TARGET_DISK)
    x_min, x_max = x_coords[0], x_coords[-1]

    if target_type == config.TARGET_LINEAR:
        # Linear profile along X
        # Линейный профиль вдоль X
        profile_coords_label = "x_position_mm"
        x_bin_edges = np.linspace(x_min, x_max, bins + 1)
        for j in range(bins):
            mask = (x_coords >= x_bin_edges[j]) & (x_coords < x_bin_edges[j+1])
            if np.any(mask):
                 profile[j] = np.mean(profile_data_source[:, mask])
            else:
                profile[j] = 0
        profile_coords = (x_bin_edges[:-1] + x_bin_edges[1:]) / 2
    else:
        # Radial profile
        # Радиальный профиль
        profile_coords_label = "radius_mm"
        profile_size = np.max(radius_grid)
        radius_bin_edges = np.linspace(0, profile_size, bins + 1)
        for j in range(bins):
            mask = (radius_grid >= radius_bin_edges[j]) & (radius_grid < radius_bin_edges[j+1])
            vals = profile_data_source[mask]
            if vals.size > 0:
                profile[j] = np.mean(vals)
            else:
                profile[j] = 0
        profile_coords = (radius_bin_edges[:-1] + radius_bin_edges[1:]) / 2

    # Suggest filename for the profile
    # Предлагаем имя файла для профиля
    profile_filename_suggestion = f'coverage_profile_{now}.csv'
    # Try to place it in the same directory as the map file if selected
    # Пытаемся поместить его в ту же директорию, что и файл карты, если он был выбран
    initial_dir = os.path.dirname(fname_map) if fname_map else "."

    fname_profile = filedialog.asksaveasfilename(
        defaultextension='.csv',
        initialdir=initial_dir,
        initialfile=profile_filename_suggestion,
        title="Сохранить профиль покрытия как CSV",
        filetypes=[('CSV файлы', '*.csv'), ('Все файлы', '*.*')]
    )

    if fname_profile:
        try:
            # Combine coordinates and profile data
            # Объединяем координаты и данные профиля
            profile_export_data = np.column_stack((profile_coords, profile))
            # Create header string
            # Создаем строку заголовка
            profile_value_label = profile_header.split(',')[1] # Get the value label (coverage_percent or particle_count)
                                                               # Получаем метку значения (coverage_percent или particle_count)
            csv_header = f"{profile_coords_label},{profile_value_label}"
            # Save profile data
            # Сохраняем данные профиля
            np.savetxt(fname_profile, profile_export_data, delimiter=',', header=csv_header, comments='', fmt='%.6g') # Use general format
                                                                                                                        # Используем общий формат
            print(f"Профиль покрытия успешно сохранен в {fname_profile}")
        except Exception as e:
            print(f"Ошибка при сохранении профиля покрытия: {e}")
            # Consider showing error to user via messagebox if in GUI context
            # Рассмотреть возможность показа ошибки пользователю через messagebox, если в контексте GUI

