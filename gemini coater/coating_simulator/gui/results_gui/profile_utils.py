# coding: utf-8
# Файл: coating_simulator/gui/results_gui/profile_utils.py
"""
Вспомогательные функции для работы с профилями покрытия в окне результатов.
Учитывают ROI при расчете статистики.
Улучшено применение ROI к диагоналям и обработка количества профилей.
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
    """
    Извлекает значения вдоль главной или побочной диагонали.
    Координаты возвращаются как индексы вдоль диагонали.
    """
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

def _apply_roi_to_profile_values(
    profile_indices_or_coords: np.ndarray, 
    profile_values: np.ndarray,
    roi_settings: dict | None,
    profile_orientation: str, 
    x_coords_centers: np.ndarray | None, 
    y_coords_centers: np.ndarray | None, 
    target_type: str,
    num_map_rows: int, # Количество строк на карте, нужно для диагоналей
    num_map_cols: int  # Количество столбцов на карте, нужно для диагоналей
    ) -> np.ndarray:
    """
    Применяет ROI к значениям профиля, заменяя значения вне ROI на np.nan.
    """
    if not roi_settings or not roi_settings.get('enabled', False) or profile_values is None:
        return profile_values

    roi_type = roi_settings.get('type')
    roi_params = roi_settings.get('params', {})
    modified_profile_values = profile_values.copy().astype(float) 

    if roi_type == 'circular' and roi_params:
        d_min = roi_params.get('d_min')
        d_max = roi_params.get('d_max')
        r_min = d_min / 2.0 if d_min is not None else 0.0
        r_max = d_max / 2.0 if d_max is not None else float('inf')

        actual_radii = np.full_like(profile_indices_or_coords, np.nan, dtype=float)

        if profile_orientation == 'horizontal' or profile_orientation == 'vertical':
            # profile_indices_or_coords это X или Y от центра (0,0)
            actual_radii = np.abs(profile_indices_or_coords) 
        elif profile_orientation in ['diag1', 'diag2']:
            if x_coords_centers is not None and y_coords_centers is not None and \
               len(profile_indices_or_coords) > 0:
                
                indices = profile_indices_or_coords.astype(int) # Это индексы от 0 до diag_len-1

                if profile_orientation == 'diag1':
                    # Главная диагональ: map_idx_row = k, map_idx_col = k
                    # Координаты центра ячейки: (x_coords_centers[map_idx_col], y_coords_centers[map_idx_row])
                    # Здесь map_idx_row = indices, map_idx_col = indices
                    map_idx_row = indices
                    map_idx_col = indices
                    
                    # Маска для валидных индексов относительно размеров карты
                    valid_mask = (map_idx_row >= 0) & (map_idx_row < num_map_rows) & \
                                 (map_idx_col >= 0) & (map_idx_col < num_map_cols)
                    
                    # Маска для валидных индексов относительно размеров массивов координат центров
                    valid_coord_mask = (map_idx_row[valid_mask] < len(y_coords_centers)) & \
                                       (map_idx_col[valid_mask] < len(x_coords_centers))

                    final_valid_mask = valid_mask.copy()
                    final_valid_mask[valid_mask] = valid_coord_mask # Обновляем только там, где valid_mask был True

                    if np.any(final_valid_mask):
                        actual_radii[final_valid_mask] = np.hypot(
                            x_coords_centers[map_idx_col[final_valid_mask]],
                            y_coords_centers[map_idx_row[final_valid_mask]]
                        )

                elif profile_orientation == 'diag2':
                    # Побочная диагональ: map_idx_row = k, map_idx_col = num_map_cols - 1 - k
                    map_idx_row = indices
                    map_idx_col = num_map_cols - 1 - indices

                    valid_mask = (map_idx_row >= 0) & (map_idx_row < num_map_rows) & \
                                 (map_idx_col >= 0) & (map_idx_col < num_map_cols)

                    valid_coord_mask = (map_idx_row[valid_mask] < len(y_coords_centers)) & \
                                       (map_idx_col[valid_mask] < len(x_coords_centers))
                    
                    final_valid_mask = valid_mask.copy()
                    final_valid_mask[valid_mask] = valid_coord_mask

                    if np.any(final_valid_mask):
                        actual_radii[final_valid_mask] = np.hypot(
                            x_coords_centers[map_idx_col[final_valid_mask]],
                            y_coords_centers[map_idx_row[final_valid_mask]]
                        )
            else: 
                return modified_profile_values 
        
        mask_out_of_roi = (actual_radii < r_min) | (actual_radii > r_max) | np.isnan(actual_radii)
        modified_profile_values[mask_out_of_roi] = np.nan
            
    elif roi_type == 'rectangular' and roi_params and target_type == config.TARGET_LINEAR:
        roi_w = roi_params.get('width')
        roi_h = roi_params.get('height')
        roi_ox = roi_params.get('offset_x', 0.0)
        roi_oy = roi_params.get('offset_y', 0.0)

        if roi_w is not None and roi_h is not None:
            half_w = roi_w / 2.0; half_h = roi_h / 2.0
            x_min_roi = roi_ox - half_w; x_max_roi = roi_ox + half_w
            y_min_roi = roi_oy - half_h; y_max_roi = roi_oy + half_h

            if profile_orientation == 'horizontal': 
                mask_out_of_roi = (profile_indices_or_coords < x_min_roi) | (profile_indices_or_coords > x_max_roi)
                modified_profile_values[mask_out_of_roi] = np.nan
            elif profile_orientation == 'vertical': 
                mask_out_of_roi = (profile_indices_or_coords < y_min_roi) | (profile_indices_or_coords > y_max_roi)
                modified_profile_values[mask_out_of_roi] = np.nan
    
    return modified_profile_values


def extract_profiles_for_statistics(
    coverage_map: np.ndarray,
    x_coords_centers: np.ndarray,
    y_coords_centers: np.ndarray,
    x_coords_edges: np.ndarray,
    y_coords_edges: np.ndarray,
    target_type: str,
    profile_config_params: dict,
    roi_settings: dict | None = None 
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
        num_profiles_requested = profile_config_params.get('num_circular_profiles', 1)
        center_y_idx = num_rows // 2
        center_x_idx = num_cols // 2
        map_center_x = 0.0; map_center_y = 0.0 
        
        temp_profiles_for_stats = [] 

        if has_x_centers:
            profile_h_data_raw = coverage_map[center_y_idx, :]
            profile_h_data_roi = _apply_roi_to_profile_values(x_coords_centers, profile_h_data_raw, roi_settings, 'horizontal', x_coords_centers, y_coords_centers, target_type, num_rows, num_cols)
            temp_profiles_for_stats.append({'coords': x_coords_centers, 'values': profile_h_data_roi, 'label': 'X (гориз.)', 'raw_values': profile_h_data_raw})
        
        if has_y_centers:
            profile_v_data_raw = coverage_map[:, center_x_idx]
            profile_v_data_roi = _apply_roi_to_profile_values(y_coords_centers, profile_v_data_raw, roi_settings, 'vertical', x_coords_centers, y_coords_centers, target_type, num_rows, num_cols)
            temp_profiles_for_stats.append({'coords': y_coords_centers, 'values': profile_v_data_roi, 'label': 'Y (верт.)', 'raw_values': profile_v_data_raw})

        if min(num_rows, num_cols) > 1:
            diag_indices1, diag_values1_raw = get_diagonal_profile(coverage_map, main_diagonal=True)
            diag_values1_roi = _apply_roi_to_profile_values(diag_indices1, diag_values1_raw, roi_settings, 'diag1', x_coords_centers, y_coords_centers, target_type, num_rows, num_cols)
            if diag_values1_raw.size > 1: temp_profiles_for_stats.append({'coords': diag_indices1, 'values': diag_values1_roi, 'label': 'Диаг. 1', 'raw_values': diag_values1_raw})
        
            diag_indices2, diag_values2_raw = get_diagonal_profile(coverage_map, main_diagonal=False)
            diag_values2_roi = _apply_roi_to_profile_values(diag_indices2, diag_values2_raw, roi_settings, 'diag2', x_coords_centers, y_coords_centers, target_type, num_rows, num_cols)
            if diag_values2_raw.size > 1: temp_profiles_for_stats.append({'coords': diag_indices2, 'values': diag_values2_roi, 'label': 'Диаг. 2', 'raw_values': diag_values2_raw})
        
        profiles_for_stats = temp_profiles_for_stats[:min(num_profiles_requested, len(temp_profiles_for_stats))]
        
        if num_profiles_requested > len(profiles_for_stats) and len(temp_profiles_for_stats) < num_profiles_requested :
             print(f"Предупреждение: Запрошено {num_profiles_requested} профилей, но для статистики используется {len(profiles_for_stats)} (макс. X, Y, Диаг1, Диаг2).")

        if temp_profiles_for_stats:
            display_profile_coords = temp_profiles_for_stats[0]['coords']
            display_profile_values = temp_profiles_for_stats[0]['raw_values'] 
            if temp_profiles_for_stats[0]['label'] == 'X (гориз.)': display_profile_axis_label = "Позиция X (мм)"
            elif temp_profiles_for_stats[0]['label'] == 'Y (верт.)': display_profile_axis_label = "Позиция Y (мм)"
            elif temp_profiles_for_stats[0]['label'] == 'Диаг. 1': display_profile_axis_label = "Индекс по диаг. 1"
            elif temp_profiles_for_stats[0]['label'] == 'Диаг. 2': display_profile_axis_label = "Индекс по диаг. 2"

        max_line_radius = 0
        if x_coords_centers is not None and y_coords_centers is not None and len(x_coords_centers)>0 and len(y_coords_centers)>0:
             max_line_radius = min(abs(x_coords_centers[-1] - x_coords_centers[0]), abs(y_coords_centers[-1] - y_coords_centers[0])) / 2.0
        
        if num_profiles_requested == 1: 
            lines_to_draw_on_map.append([(map_x_min, map_center_y), (map_x_max, map_center_y)])
        elif num_profiles_requested > 1 and max_line_radius > 0:
            for i in range(num_profiles_requested): 
                angle_rad = i * (math.pi / num_profiles_requested)
                x_start = map_center_x - max_line_radius * math.cos(angle_rad)
                y_start = map_center_y - max_line_radius * math.sin(angle_rad)
                x_end = map_center_x + max_line_radius * math.cos(angle_rad)
                y_end = map_center_y + max_line_radius * math.sin(angle_rad)
                lines_to_draw_on_map.append([(x_start, y_start), (x_end, y_end)])

    elif target_type == config.TARGET_LINEAR:
        profile_type_config = profile_config_params.get('linear_profile_type', 'horizontal')
        y_offset_for_x_profile = profile_config_params.get('linear_x_profile_y_offset', 0.0)
        x_offset_for_y_profile = profile_config_params.get('linear_y_profile_x_offset', 0.0)

        if profile_type_config in ['horizontal', 'both'] and has_x_centers and has_y_centers:
            iy = _find_nearest_idx(y_coords_centers, y_offset_for_x_profile)
            actual_y_coord = y_coords_centers[iy]
            profile_x_data_raw = coverage_map[iy, :]
            profile_x_data_roi = _apply_roi_to_profile_values(x_coords_centers, profile_x_data_raw, roi_settings, 'horizontal', x_coords_centers, y_coords_centers, target_type, num_rows, num_cols)
            profiles_for_stats.append({
                'coords': x_coords_centers, 'values': profile_x_data_roi,
                'label': f'X (Y={actual_y_coord:.1f})'
            })
            lines_to_draw_on_map.append([(map_x_min, actual_y_coord), (map_x_max, actual_y_coord)])
            if display_profile_values is None:
                display_profile_coords = x_coords_centers; display_profile_values = profile_x_data_raw
                display_profile_axis_label = f"Позиция X (при Y={actual_y_coord:.1f} мм)"

        if profile_type_config in ['vertical', 'both'] and has_y_centers and has_x_centers:
            ix = _find_nearest_idx(x_coords_centers, x_offset_for_y_profile)
            actual_x_coord = x_coords_centers[ix]
            profile_y_data_raw = coverage_map[:, ix]
            profile_y_data_roi = _apply_roi_to_profile_values(y_coords_centers, profile_y_data_raw, roi_settings, 'vertical', x_coords_centers, y_coords_centers, target_type, num_rows, num_cols)
            profiles_for_stats.append({
                'coords': y_coords_centers, 'values': profile_y_data_roi,
                'label': f'Y (X={actual_x_coord:.1f})'
            })
            lines_to_draw_on_map.append([(actual_x_coord, map_y_min), (actual_x_coord, map_y_max)])
            if display_profile_values is None: 
                display_profile_coords = y_coords_centers; display_profile_values = profile_y_data_raw
                display_profile_axis_label = f"Позиция Y (при X={actual_x_coord:.1f} мм)"
    else: 
        if has_x_centers:
            center_y_idx_map = num_rows // 2
            actual_y_coord_fallback = y_coords_centers[center_y_idx_map] if has_y_centers else 0.0
            profile_data_raw = coverage_map[center_y_idx_map, :]
            profile_data_roi = _apply_roi_to_profile_values(x_coords_centers, profile_data_raw, roi_settings, 'horizontal', x_coords_centers, y_coords_centers, target_type, num_rows, num_cols)
            profiles_for_stats.append({'coords': x_coords_centers, 'values': profile_data_roi, 'label': 'X (центр)'})
            lines_to_draw_on_map.append([(map_x_min, actual_y_coord_fallback), 
                                         (map_x_max, actual_y_coord_fallback)])
            if display_profile_values is None:
                display_profile_coords = x_coords_centers; display_profile_values = profile_data_raw
                display_profile_axis_label = "Позиция X (мм)"

    if display_profile_values is None and profiles_for_stats: 
        first_profile_dict = profiles_for_stats[0]
        display_profile_coords = first_profile_dict['coords']
        display_profile_values = first_profile_dict.get('raw_values', first_profile_dict['values'])
        
        label_from_dict = first_profile_dict.get('label', '')
        if 'X' in label_from_dict: display_profile_axis_label = f"Позиция X ({label_from_dict})"
        elif 'Y' in label_from_dict: display_profile_axis_label = f"Позиция Y ({label_from_dict})"
        elif 'Диаг' in label_from_dict: display_profile_axis_label = f"Индекс по диаг. ({label_from_dict})"

    return profiles_for_stats, display_profile_coords, display_profile_values, display_profile_axis_label, lines_to_draw_on_map


def calculate_and_format_uniformity(
    profiles_for_stats: list[dict],
    view_settings: dict,
    coverage_map_for_norm: np.ndarray | None = None,
    roi_settings: dict | None = None 
) -> tuple[str, dict | None]:
    # ... (логика без изменений) ...
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
        values_for_stats = prof_data_dict.get('values') 

        if values_for_stats is None or coords is None or \
           len(coords) != len(values_for_stats) or len(values_for_stats) < 2:
            uniformity_texts.append(f"U ({prof_data_dict.get('label', '?')}): Нет данных")
            continue
        
        values_for_processing_stats = values_for_stats.copy() 
        if display_percent: 
            values_for_processing_stats = values_for_processing_stats / max_val_for_norm * 100.0
        
        if not np.any(np.isfinite(values_for_processing_stats)):
            uniformity_texts.append(f"U ({prof_data_dict.get('label', '?')}): Нет валидных данных (в ROI)")
            if i == 0: first_profile_processed_stats_for_gui = {'t_max': "-", 't_min': "-", 't_mean': "-"}
            continue
        
        data_for_stats_calculation = values_for_processing_stats
        if not calculate_stats_on_raw: 
            smoothed_data = smooth_profile_data(coords, values_for_processing_stats, smoothing_method, smoothing_params_dict)
            data_for_stats_calculation = smoothed_data
        
        stats_current = calculate_uniformity_stats(data_for_stats_calculation) 
        u_text = format_uniformity_stats(stats_current, uniformity_method_key)
        u_text_labeled = u_text.replace(f"{uniformity_method_key}=", f"{uniformity_method_key} ({prof_data_dict.get('label', '?')})=")
        uniformity_texts.append(u_text_labeled)
        
        if i == 0: 
            first_profile_processed_stats_for_gui = {
                't_max': f"{np.nanmax(data_for_stats_calculation[np.isfinite(data_for_stats_calculation)]):.2f}" if np.any(np.isfinite(data_for_stats_calculation)) else "-",
                't_min': f"{np.nanmin(data_for_stats_calculation[np.isfinite(data_for_stats_calculation)]):.2f}" if np.any(np.isfinite(data_for_stats_calculation)) else "-",
                't_mean': f"{np.nanmean(data_for_stats_calculation[np.isfinite(data_for_stats_calculation)]):.2f}" if np.any(np.isfinite(data_for_stats_calculation)) else "-"
            }
    uniformity_result_text = "\n".join(uniformity_texts) if uniformity_texts else "U: Нет данных для расчета"
    return uniformity_result_text, first_profile_processed_stats_for_gui

if __name__ == '__main__':
    # ... (тестовый код без изменений) ...
    print("Тестирование profile_utils.py (v9 - ROI diagonals attempt)")
    mock_map = np.array([
        [10, 12, 15, 12, 10],
        [12, 20, 25, 20, 12],
        [15, 25, 30, 25, 15], 
        [12, 20, 25, 20, 12],
        [10, 12, 15, 12, 10]
    ]).astype(float)
    mock_x_c = np.array([-2., -1.,  0.,  1.,  2.]) # Центры ячеек по X
    mock_y_c = np.array([-2., -1.,  0.,  1.,  2.]) # Центры ячеек по Y
    mock_x_e = np.array([-2.5, -1.5, -0.5,  0.5,  1.5,  2.5])
    mock_y_e = np.array([-2.5, -1.5, -0.5,  0.5,  1.5,  2.5])

    print("\n--- Тест для КРУГЛОЙ мишени (статистика с ROI) ---")
    mock_roi_circular = {'enabled': True, 'show_on_map': True, 'type': 'circular', 
                         'params': {'d_min': 1.0, 'd_max': 3.0}} # r_min=0.5, r_max=1.5
    
    for num_req in [1, 2, 3, 4, 5, 6]:
        circular_params_cfg = {'num_circular_profiles': num_req} 
        p_stats_circ, dpc, dpv_raw, dpal, lines = extract_profiles_for_statistics(
            mock_map, mock_x_c, mock_y_c, mock_x_e, mock_y_e, config.TARGET_DISK, circular_params_cfg, mock_roi_circular
        )
        print(f"\nКруг (ROI D=[1,3]), запрошено {num_req} профилей:")
        print(f"  Извлечено для стат.: {len(p_stats_circ)}")
        for p_s in p_stats_circ:
             print(f"    Профиль '{p_s['label']}' (коорд.): {p_s['coords']}")
             print(f"    Профиль '{p_s['label']}' (после ROI): {p_s['values']}")
             print(f"    Профиль '{p_s['label']}' (сырой): {p_s.get('raw_values')}")

        
        mock_view_settings_circ = {
            'profile_config_params': circular_params_cfg, 'uniformity_method_key': 'U3',
            'smoothing': {'method': "Без сглаживания", 'params': {}},
            'display_percent': False, 'show_raw_profile': True, 'roi': mock_roi_circular
        }
        uniformity_text_circ, stats_gui = calculate_and_format_uniformity(p_stats_circ, mock_view_settings_circ, mock_map, mock_roi_circular)
        print(f"  Результат U (с ROI) ({len(uniformity_text_circ.splitlines())} строк):\n    {uniformity_text_circ.replace(chr(10), chr(10) + chr(32)*4)}")

    print("\nТестирование profile_utils.py (v9) завершено.")
