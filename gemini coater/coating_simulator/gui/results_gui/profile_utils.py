# coding: utf-8
# Файл: coating_simulator/gui/results_gui/profile_utils.py
"""
Вспомогательные функции для работы с профилями покрытия в окне результатов.
Исправлено: количество отображаемых результатов равномерности теперь соответствует
количеству извлеченных профилей для статистики.
"""
import numpy as np
import math

try:
    from ...visualization.plot import (calculate_uniformity_stats,
                                       format_uniformity_stats,
                                       _smooth_profile as smooth_profile_data)
    from ... import config
    PLOT_MODULE_AVAILABLE = True
except ImportError:
    print("ОШИБКА ИМПОРТА (profile_utils.py): Не удалось импортировать из visualization.plot или config.")
    PLOT_MODULE_AVAILABLE = False
    def calculate_uniformity_stats(*args, **kwargs): return {}
    def format_uniformity_stats(*args, **kwargs): return "Ошибка импорта plot.py"
    def smooth_profile_data(coords, data, method, params): return data
    class ConfigMock: #type: ignore
        TARGET_DISK = "диск"; TARGET_DOME = "купол"; TARGET_PLANETARY = "планетарный"
        TARGET_LINEAR = "линейное перемещение"
    config = ConfigMock()


def get_diagonal_profile(coverage_map: np.ndarray, main_diagonal: bool = True) -> tuple[np.ndarray, np.ndarray]:
    if coverage_map is None or coverage_map.ndim != 2:
        return np.array([]), np.array([])
    rows, cols = coverage_map.shape
    diag_len = min(rows, cols)
    if diag_len == 0: return np.array([]), np.array([])
    if main_diagonal:
        profile_values = np.array([coverage_map[i, i] for i in range(diag_len)])
    else:
        profile_values = np.array([coverage_map[i, cols - 1 - i] for i in range(diag_len)])
    profile_coords = np.arange(diag_len)
    return profile_coords, profile_values

def _find_nearest_idx(array_coords: np.ndarray, value: float) -> int:
    if array_coords is None or len(array_coords) == 0:
        return 0
    idx = np.abs(array_coords - value).argmin()
    return int(idx)

def extract_profiles_for_statistics(
    coverage_map: np.ndarray,
    x_coords_centers: np.ndarray,
    y_coords_centers: np.ndarray,
    x_coords_edges: np.ndarray,
    y_coords_edges: np.ndarray,
    target_type: str,
    profile_config_params: dict
) -> tuple[list[dict], np.ndarray | None, np.ndarray | None, str, list[list[tuple[float, float]]]]:
    profiles_for_stats: list[dict] = []
    display_profile_coords: np.ndarray | None = None
    display_profile_values: np.ndarray | None = None
    display_profile_axis_label: str = "Позиция (мм)"
    lines_to_draw_on_map: list[list[tuple[float, float]]] = []

    if coverage_map is None or coverage_map.ndim != 2:
        return profiles_for_stats, display_profile_coords, display_profile_values, display_profile_axis_label, lines_to_draw_on_map

    num_rows, num_cols = coverage_map.shape
    has_x_centers = x_coords_centers is not None and len(x_coords_centers) == num_cols and num_cols > 0
    has_y_centers = y_coords_centers is not None and len(y_coords_centers) == num_rows and num_rows > 0
    
    map_x_min = x_coords_edges[0] if x_coords_edges is not None and len(x_coords_edges)>0 else 0
    map_x_max = x_coords_edges[-1] if x_coords_edges is not None and len(x_coords_edges)>0 else 0
    map_y_min = y_coords_edges[0] if y_coords_edges is not None and len(y_coords_edges)>0 else 0
    map_y_max = y_coords_edges[-1] if y_coords_edges is not None and len(y_coords_edges)>0 else 0

    if target_type in [config.TARGET_DISK, config.TARGET_DOME, config.TARGET_PLANETARY]:
        num_profiles_for_map_lines = profile_config_params.get('num_circular_profiles', 1)
        center_y_idx = num_rows // 2
        center_x_idx = num_cols // 2
        map_center_x = 0.0; map_center_y = 0.0
        max_line_radius = min(abs(map_x_max - map_x_min), abs(map_y_max - map_y_min)) / 2.0
        
        # Определяем, сколько профилей ИЗВЛЕЧЬ для СТАТИСТИКИ
        # Максимум 4: X, Y, Диаг1, Диаг2. Это не зависит напрямую от num_profiles_for_map_lines,
        # если num_profiles_for_map_lines > 4. Но если < 4, то извлекаем меньше.
        num_profiles_for_stats_calc = min(num_profiles_for_map_lines, 4)


        if num_profiles_for_stats_calc >= 1 and has_x_centers:
            profile_h_data = coverage_map[center_y_idx, :]
            profiles_for_stats.append({'coords': x_coords_centers, 'values': profile_h_data, 'label': 'X (гориз.)'})
            if display_profile_values is None:
                display_profile_coords = x_coords_centers; display_profile_values = profile_h_data
                display_profile_axis_label = "Позиция X (мм)"
        if num_profiles_for_stats_calc >= 2 and has_y_centers:
            profile_v_data = coverage_map[:, center_x_idx]
            profiles_for_stats.append({'coords': y_coords_centers, 'values': profile_v_data, 'label': 'Y (верт.)'})
            if display_profile_values is None:
                display_profile_coords = y_coords_centers; display_profile_values = profile_v_data
                display_profile_axis_label = "Позиция Y (мм)"
        if num_profiles_for_stats_calc >= 3 and min(num_rows, num_cols) > 1:
            diag_coords1, diag_values1 = get_diagonal_profile(coverage_map, main_diagonal=True)
            if diag_values1.size > 1: profiles_for_stats.append({'coords': diag_coords1, 'values': diag_values1, 'label': 'Диаг. 1'})
            if display_profile_values is None and diag_values1.size > 1:
                display_profile_coords = diag_coords1; display_profile_values = diag_values1
                display_profile_axis_label = "Позиция по диаг. 1 (индекс)"
        if num_profiles_for_stats_calc >= 4 and min(num_rows, num_cols) > 1:
            diag_coords2, diag_values2 = get_diagonal_profile(coverage_map, main_diagonal=False)
            if diag_values2.size > 1: profiles_for_stats.append({'coords': diag_coords2, 'values': diag_values2, 'label': 'Диаг. 2'})
            if display_profile_values is None and diag_values2.size > 1:
                display_profile_coords = diag_coords2; display_profile_values = diag_values2
                display_profile_axis_label = "Позиция по диаг. 2 (индекс)"

        # Линии для отрисовки на карте (всегда num_profiles_for_map_lines штук)
        if num_profiles_for_map_lines == 1:
            lines_to_draw_on_map.append([(map_x_min, map_center_y), (map_x_max, map_center_y)])
        elif num_profiles_for_map_lines > 1:
            for i in range(num_profiles_for_map_lines):
                angle_rad = i * (math.pi / num_profiles_for_map_lines)
                x_start = map_center_x - max_line_radius * math.cos(angle_rad)
                y_start = map_center_y - max_line_radius * math.sin(angle_rad)
                x_end = map_center_x + max_line_radius * math.cos(angle_rad)
                y_end = map_center_y + max_line_radius * math.sin(angle_rad)
                lines_to_draw_on_map.append([(x_start, y_start), (x_end, y_end)])

    elif target_type == config.TARGET_LINEAR:
        y_offset_for_x_profile = profile_config_params.get('linear_x_profile_y_offset', 0.0)
        x_offset_for_y_profile = profile_config_params.get('linear_y_profile_x_offset', 0.0)

        if has_x_centers and has_y_centers:
            iy = _find_nearest_idx(y_coords_centers, y_offset_for_x_profile)
            actual_y_coord = y_coords_centers[iy]
            profile_x_data = coverage_map[iy, :]
            profiles_for_stats.append({
                'coords': x_coords_centers, 'values': profile_x_data,
                'label': f'X (Y={actual_y_coord:.1f})'
            })
            lines_to_draw_on_map.append([(map_x_min, actual_y_coord), (map_x_max, actual_y_coord)])
            if display_profile_values is None:
                display_profile_coords = x_coords_centers; display_profile_values = profile_x_data
                display_profile_axis_label = f"Позиция X (при Y={actual_y_coord:.1f} мм)"

        if has_y_centers and has_x_centers:
            ix = _find_nearest_idx(x_coords_centers, x_offset_for_y_profile)
            actual_x_coord = x_coords_centers[ix]
            profile_y_data = coverage_map[:, ix]
            profiles_for_stats.append({
                'coords': y_coords_centers, 'values': profile_y_data,
                'label': f'Y (X={actual_x_coord:.1f})'
            })
            lines_to_draw_on_map.append([(actual_x_coord, map_y_min), (actual_x_coord, map_y_max)])
            if display_profile_values is None:
                display_profile_coords = y_coords_centers; display_profile_values = profile_y_data
                display_profile_axis_label = f"Позиция Y (при X={actual_x_coord:.1f} мм)"
    else:
        if has_x_centers:
            center_y_idx_map = num_rows // 2
            actual_y_coord_fallback = y_coords_centers[center_y_idx_map] if has_y_centers else 0.0
            profile_data = coverage_map[center_y_idx_map, :]
            profiles_for_stats.append({'coords': x_coords_centers, 'values': profile_data, 'label': 'X (центр)'})
            lines_to_draw_on_map.append([(map_x_min, actual_y_coord_fallback), 
                                         (map_x_max, actual_y_coord_fallback)])
            if display_profile_values is None:
                display_profile_coords = x_coords_centers; display_profile_values = profile_data
                display_profile_axis_label = "Позиция X (мм)"

    if display_profile_values is None and profiles_for_stats:
        first_profile_dict = profiles_for_stats[0]
        display_profile_coords = first_profile_dict['coords']
        display_profile_values = first_profile_dict['values']
        label_from_dict = first_profile_dict.get('label', '')
        if 'X' in label_from_dict: display_profile_axis_label = f"Позиция X ({label_from_dict})"
        elif 'Y' in label_from_dict: display_profile_axis_label = f"Позиция Y ({label_from_dict})"
        elif 'Диаг' in label_from_dict: display_profile_axis_label = f"Позиция по диаг. ({label_from_dict})"

    return profiles_for_stats, display_profile_coords, display_profile_values, display_profile_axis_label, lines_to_draw_on_map


def calculate_and_format_uniformity(
    profiles_for_stats: list[dict],
    view_settings: dict,
    coverage_map_for_norm: np.ndarray | None = None
) -> tuple[str, dict | None]:
    # ... (остальная часть функции без изменений, она уже корректно обрабатывает список profiles_for_stats)
    if not PLOT_MODULE_AVAILABLE: return "Ошибка: Модуль plot.py недоступен.", None
    smoothing_method = view_settings.get('smoothing', {}).get('method', "Без сглаживания")
    smoothing_params_dict = view_settings.get('smoothing', {}).get('params', {})
    display_percent = view_settings.get('display_percent', True)
    calculate_stats_on_raw = view_settings.get('show_raw_profile', False) or smoothing_method == "Без сглаживания"
    uniformity_method_key = view_settings.get('uniformity_method_key', 'U3')

    uniformity_texts = []
    first_profile_processed_stats_for_gui = None
    if not profiles_for_stats: return "U: Нет профилей для расчета", None

    max_val_for_norm = 1.0
    if display_percent and coverage_map_for_norm is not None:
        max_val_for_norm_map_raw = np.nanmax(coverage_map_for_norm.astype(float))
        if max_val_for_norm_map_raw is not None and max_val_for_norm_map_raw > 0:
            max_val_for_norm = max_val_for_norm_map_raw

    for i, prof_data_dict in enumerate(profiles_for_stats):
        coords = prof_data_dict.get('coords')
        values_raw_unnormalized = prof_data_dict.get('values')
        if values_raw_unnormalized is None or coords is None or \
           len(coords) != len(values_raw_unnormalized) or len(values_raw_unnormalized) < 2:
            uniformity_texts.append(f"U ({prof_data_dict.get('label', '?')}): Нет данных")
            continue
        values_for_processing = values_raw_unnormalized.copy()
        if display_percent:
            values_for_processing = values_for_processing / max_val_for_norm * 100.0
        if not np.any(np.isfinite(values_for_processing)):
            uniformity_texts.append(f"U ({prof_data_dict.get('label', '?')}): Нет валидных данных")
            if i == 0: first_profile_processed_stats_for_gui = {'t_max': "-", 't_min': "-", 't_mean': "-"}
            continue
        data_for_stats_calculation = values_for_processing
        if not calculate_stats_on_raw:
            smoothed_data = smooth_profile_data(coords, values_for_processing, smoothing_method, smoothing_params_dict)
            data_for_stats_calculation = smoothed_data
        stats_current = calculate_uniformity_stats(data_for_stats_calculation)
        u_text = format_uniformity_stats(stats_current, uniformity_method_key)
        u_text_labeled = u_text.replace(f"{uniformity_method_key}=", f"{uniformity_method_key} ({prof_data_dict.get('label', '?')})=")
        uniformity_texts.append(u_text_labeled)
        if i == 0:
            first_profile_processed_stats_for_gui = {
                't_max': f"{np.nanmax(data_for_stats_calculation):.2f}" if np.any(np.isfinite(data_for_stats_calculation)) else "-",
                't_min': f"{np.nanmin(data_for_stats_calculation):.2f}" if np.any(np.isfinite(data_for_stats_calculation)) else "-",
                't_mean': f"{np.nanmean(data_for_stats_calculation):.2f}" if np.any(np.isfinite(data_for_stats_calculation)) else "-"
            }
    uniformity_result_text = "\n".join(uniformity_texts) if uniformity_texts else "U: Нет данных для расчета"
    return uniformity_result_text, first_profile_processed_stats_for_gui


if __name__ == '__main__':
    print("Тестирование profile_utils.py (v4 - uniformity display fix)")
    # ... (остальной тестовый код без изменений) ...
    mock_map = np.arange(25).reshape(5,5).astype(float)
    mock_x_c = np.array([-2., -1.,  0.,  1.,  2.])
    mock_y_c = np.array([-2., -1.,  0.,  1.,  2.])
    mock_x_e = np.array([-2.5, -1.5, -0.5,  0.5,  1.5,  2.5])
    mock_y_e = np.array([-2.5, -1.5, -0.5,  0.5,  1.5,  2.5])

    print("\n--- Тест для КРУГЛОЙ мишени (статистика) ---")
    for num_p_stat in [1, 2, 3, 4, 5, 6]:
        circular_params = {'num_circular_profiles': num_p_stat}
        p_stats, _, _, _, lines = extract_profiles_for_statistics(
            mock_map, mock_x_c, mock_y_c, mock_x_e, mock_y_e, config.TARGET_DISK, circular_params
        )
        print(f"Круг (запрошено {num_p_stat} линий на карте):")
        print(f"  Профилей для статистики: {len(p_stats)}")
        for p_s in p_stats: print(f"    - {p_s['label']}")
        print(f"  Линий на карте: {len(lines)}")
        
        mock_view_settings = {
            'profile_config_params': circular_params,
            'uniformity_method_key': 'U3',
            'smoothing': {'method': "Без сглаживания", 'params': {}},
            'display_percent': False, 'show_raw_profile': True
        }
        uniformity_text, _ = calculate_and_format_uniformity(p_stats, mock_view_settings, mock_map)
        print(f"  Результат U (для {len(p_stats)} профилей):\n    {uniformity_text.replace(chr(10), chr(10) + chr(32)*4)}")
    print("\nТестирование profile_utils.py (v4) завершено.")
