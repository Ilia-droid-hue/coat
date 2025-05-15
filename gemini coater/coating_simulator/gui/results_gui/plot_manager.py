# coding: utf-8
# Файл: coating_simulator/gui/results_gui/plot_manager.py
"""
Управляет обновлением и отображением графиков в окне результатов.
Отвечает за вызов функций отрисовки из модуля visualization.plot.
Теперь передает координаты линий профилей и настройки ROI для отрисовки на карте.
"""
import numpy as np
import traceback

try:
    from ...visualization.plot import (plot_simulation_results as plot_on_figure,
                                       _smooth_profile as smooth_profile_data)
    from .profile_utils import PLOT_MODULE_AVAILABLE # Зависит от profile_utils для флага
except ImportError:
    print("ОШИБКА ИМПОРТА (plot_manager.py): Не удалось импортировать из visualization.plot или profile_utils.")
    PLOT_MODULE_AVAILABLE = False
    def plot_on_figure(*args, **kwargs):
        print("plot_on_figure (заглушка из plot_manager)")
    def smooth_profile_data(coords, data, method, params):
        print("smooth_profile_data (заглушка из plot_manager)")
        return data


def update_plots(
    map_figure,
    profile_figure,
    coverage_map_data: np.ndarray,
    x_coords_edges_data: np.ndarray,
    y_coords_edges_data: np.ndarray,
    radius_grid_centers_data: np.ndarray,
    target_params_data: dict,
    current_vis_params_data: dict, # Этот словарь теперь будет содержать и 'cmap' и 'roi' из view_settings
    # roi_settings_data: dict | None, # <-- УДАЛЯЕМ ЭТОТ ОТДЕЛЬНЫЙ ПАРАМЕТР
    display_profile_coords: np.ndarray | None,
    display_profile_values_processed: np.ndarray | None,
    display_profile_axis_label: str,
    lines_to_draw_on_map: list[list[tuple[float, float]]] | None = None,
    raw_profile_for_comparison_coords: np.ndarray | None = None,
    raw_profile_for_comparison_values: np.ndarray | None = None,
    raw_profile_label: str = "Сырой (сравн.)"
):
    """
    Обновляет и перерисовывает карту покрытия и график профиля.
    Настройки ROI теперь передаются внутри current_vis_params_data.
    """
    if not PLOT_MODULE_AVAILABLE:
        print("Plotting is disabled due to import errors (plot_manager).")
        if map_figure:
            map_figure.clear()
            ax_map_err = map_figure.add_subplot(111)
            ax_map_err.text(0.5, 0.5, "Ошибка загрузки модуля\nвизуализации",
                            ha='center', va='center', fontsize=10, color='red')
        if profile_figure:
            profile_figure.clear()
            ax_prof_err = profile_figure.add_subplot(111)
            ax_prof_err.text(0.5, 0.5, "Ошибка загрузки модуля\nвизуализации",
                             ha='center', va='center', fontsize=10, color='red')
        return

    try:
        # Извлекаем roi_settings из current_vis_params_data, если они там есть
        roi_settings_to_pass = current_vis_params_data.get('roi')

        # --- Обновление карты покрытия ---
        if map_figure is not None:
            map_figure.clear()
            plot_on_figure(
                fig=map_figure,
                coverage_map=coverage_map_data,
                x_coords=x_coords_edges_data,
                y_coords=y_coords_edges_data,
                radius_grid=radius_grid_centers_data,
                target_params=target_params_data,
                vis_params=current_vis_params_data, # Передаем весь словарь vis_params
                roi_settings=roi_settings_to_pass,  # Передаем извлеченные или None
                profile_lines_coords=lines_to_draw_on_map,
                show_colorbar=True,
                plot_type="map_only"
            )
            map_figure.tight_layout(pad=1.0)

        # --- Обновление графика профиля ---
        if profile_figure is not None:
            profile_figure.clear()
            ax_profile_actual = profile_figure.add_subplot(111)

            if display_profile_coords is not None and display_profile_values_processed is not None and \
               len(display_profile_coords) > 0 and len(display_profile_values_processed) > 0:

                ax_profile_actual.plot(display_profile_coords, display_profile_values_processed,
                                       '-', color='orange', linewidth=1.5, label='Профиль')

                if raw_profile_for_comparison_coords is not None and \
                   raw_profile_for_comparison_values is not None and \
                   len(raw_profile_for_comparison_coords) == len(display_profile_coords) and \
                   not np.allclose(display_profile_values_processed, raw_profile_for_comparison_values, equal_nan=True):
                    ax_profile_actual.plot(raw_profile_for_comparison_coords,
                                           raw_profile_for_comparison_values,
                                           ':', color='gray', linewidth=1.0, label=raw_profile_label)

                ax_profile_actual.set_title('Профиль покрытия', fontsize=10)
                ax_profile_actual.set_xlabel(display_profile_axis_label, fontsize=9)
                ax_profile_actual.set_ylabel(
                    'Покрытие (%)' if current_vis_params_data.get('percent', True) else 'Количество частиц',
                    fontsize=9
                )
                ax_profile_actual.grid(True, linestyle=':')
                if ax_profile_actual.has_data():
                    ax_profile_actual.legend(fontsize='small')

                valid_plot_values = display_profile_values_processed[np.isfinite(display_profile_values_processed)]
                if valid_plot_values.size > 0:
                    y_min_plot = np.nanmin(valid_plot_values)
                    y_max_plot = np.nanmax(valid_plot_values)
                    padding = (y_max_plot - y_min_plot) * 0.05 if (y_max_plot - y_min_plot) > 1e-6 else 0.1
                    final_y_min = y_min_plot - padding
                    final_y_max = y_max_plot + padding
                    if current_vis_params_data.get('percent', True):
                        final_y_min = max(0, final_y_min)
                        final_y_max = min(110, final_y_max) if final_y_max > 0 else 10
                        if final_y_max <= final_y_min: final_y_max = final_y_min + 10
                    ax_profile_actual.set_ylim(bottom=final_y_min, top=final_y_max)
                else:
                    ax_profile_actual.set_ylim(0, 1 if not current_vis_params_data.get('percent', True) else 10)
                    ax_profile_actual.text(0.5, 0.5, "Нет данных\nдля профиля",
                                           horizontalalignment='center', verticalalignment='center',
                                           transform=ax_profile_actual.transAxes, fontsize=9)
            else:
                ax_profile_actual.text(0.5, 0.5, "Нет данных\nдля профиля",
                                       horizontalalignment='center', verticalalignment='center',
                                       transform=ax_profile_actual.transAxes, fontsize=9)
                ax_profile_actual.set_title('Профиль покрытия', fontsize=10)
                ax_profile_actual.set_xlabel(display_profile_axis_label, fontsize=9)
                ax_profile_actual.set_ylabel(
                    'Покрытие (%)' if current_vis_params_data.get('percent', True) else 'Количество частиц',
                    fontsize=9
                )
                ax_profile_actual.grid(True, linestyle=':')
                ax_profile_actual.set_ylim(0, 1 if not current_vis_params_data.get('percent', True) else 10)
            profile_figure.tight_layout(pad=0.8)

    except Exception as e:
        print(f"Ошибка в plot_manager.update_plots: {e}")
        traceback.print_exc()
        if map_figure:
            try:
                map_figure.clear(); ax_map_err = map_figure.add_subplot(111)
                ax_map_err.text(0.5, 0.5, f"Ошибка карты:\n{e}", ha='center', va='center', fontsize=8, color='red', wrap=True)
            except: pass
        if profile_figure:
            try:
                profile_figure.clear(); ax_prof_err = profile_figure.add_subplot(111)
                ax_prof_err.text(0.5, 0.5, f"Ошибка профиля:\n{e}", ha='center', va='center', fontsize=8, color='red', wrap=True)
            except: pass

if __name__ == '__main__':
    print("Тестирование plot_manager.py (v3.1 - ROI in vis_params)")
    from matplotlib.figure import Figure
    mock_map_fig = Figure(); mock_profile_fig = Figure()
    mock_coverage = np.random.rand(10, 10) * 100
    mock_x_edges = np.linspace(-5, 5, 11); mock_y_edges = np.linspace(-5, 5, 11)
    mock_radius_grid = np.hypot(*np.meshgrid((mock_x_edges[:-1] + mock_x_edges[1:])/2, (mock_y_edges[:-1] + mock_y_edges[1:])/2))
    mock_target = {'target_type': 'диск', 'diameter': 10}
    
    mock_roi = {'show_on_map': True, 'type': 'circular', 'params': {'d_min': 2.0, 'd_max': 8.0}}
    mock_vis_with_roi = {'percent': True, 'logscale': False, 'roi': mock_roi, 'cmap': {}} # Добавляем ROI сюда

    mock_prof_coords = (mock_x_edges[:-1] + mock_x_edges[1:])/2
    mock_prof_values = np.mean(mock_coverage, axis=0)
    if mock_vis_with_roi['percent']: 
        max_cov = np.max(mock_coverage)
        mock_prof_values = mock_prof_values / max_cov * 100 if max_cov > 0 else mock_prof_values
    
    mock_lines = [[(-5.0, 0.0), (5.0, 0.0)], [(0.0, -5.0), (0.0, 5.0)]]

    update_plots(
        map_figure=mock_map_fig, profile_figure=mock_profile_fig,
        coverage_map_data=mock_coverage, x_coords_edges_data=mock_x_edges,
        y_coords_edges_data=mock_y_edges, radius_grid_centers_data=mock_radius_grid,
        target_params_data=mock_target, 
        current_vis_params_data=mock_vis_with_roi, # Передаем объединенные параметры
        display_profile_coords=mock_prof_coords, display_profile_values_processed=mock_prof_values,
        display_profile_axis_label="Позиция X (мм)",
        lines_to_draw_on_map=mock_lines
    )
    print("Тест update_plots с ROI в vis_params завершен.")
    # mock_map_fig.savefig("test_map_with_roi_in_vis_params.png")
