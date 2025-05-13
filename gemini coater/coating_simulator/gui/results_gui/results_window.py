# coding: utf-8
# Файл: coating_simulator/gui/results_gui/results_window.py
"""
Содержит класс ResultsWindow, основное окно для отображения результатов симуляции.
Исправлена ошибка AttributeError путем контроля вызова recalculate_callback
во время инициализации SettingsPanel.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import traceback
import math
import os

from .settings_panel import SettingsPanel
from .display_areas import PlotDisplayArea, InfoDisplayArea
from .profile_utils import (
    extract_profiles_for_statistics,
    calculate_and_format_uniformity
)
from .plot_manager import update_plots
from .action_callbacks import (
    placeholder_calculate_mask,
    placeholder_export_excel,
    placeholder_load_profiles,
    placeholder_reconstruct_map
)

try:
    from ... import config
    from ...visualization.plot import _smooth_profile as smooth_profile_data
    from ...visualization.plot import plot_simulation_results as _
    PLOT_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"ОШИБКА ИМПОРТА в results_window.py: {e}")
    PLOT_MODULE_AVAILABLE = False
    class ConfigMock: #type: ignore
        TARGET_DISK = "диск"; TARGET_DOME = "купол"; TARGET_PLANETARY = "планетарный"
        TARGET_LINEAR = "линейное перемещение"; VIS_DEFAULT_PERCENT = True
        VIS_DEFAULT_LOGSCALE = False; SIM_GRID_SIZE = 51
    config = ConfigMock()
    def smooth_profile_data(coords, data, method, params): return data

import matplotlib.pyplot as plt

class ResultsWindow(tk.Toplevel):
    def __init__(self, parent, coverage_map: np.ndarray,
                 x_coords_edges: np.ndarray, y_coords_edges: np.ndarray,
                 radius_grid_centers: np.ndarray,
                 target_params: dict, vis_params: dict):
        super().__init__(parent)
        self.title("Результаты: Анализ и Профили")
        self.geometry("1100x700")
        self.minsize(800, 600)

        self.simulation_coverage_map_raw = coverage_map
        self.simulation_x_coords_edges = x_coords_edges
        self.simulation_y_coords_edges = y_coords_edges
        self.simulation_radius_grid_centers = radius_grid_centers
        self.simulation_target_params = target_params
        self.current_vis_params = vis_params.copy()
        self.loaded_profiles_data = []
        self.current_target_type = self.simulation_target_params.get('target_type', config.TARGET_DISK)

        self.columnconfigure(0, weight=0, minsize=280)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # 1. Создаем SettingsPanel
        self.settings_panel = SettingsPanel(
            self,
            recalculate_callback=self._recalculate_and_redraw_plots,
            export_excel_callback=self._handle_export_excel,
            calculate_mask_callback=self._handle_calculate_mask,
            load_profiles_callback=self._handle_load_profiles,
            reconstruct_map_callback=self._handle_reconstruct_map
        )
        self.settings_panel.grid(row=0, column=0, sticky=tk.NSEW, padx=(10,5), pady=(10,10))

        # 2. Создаем остальные панели
        self.update_idletasks()
        try:
            s_temp = ttk.Style(); left_panel_bg_color = s_temp.lookup("TFrame", "background")
        except tk.TclError: left_panel_bg_color = "SystemButtonFace"
        if not isinstance(left_panel_bg_color, str) or not left_panel_bg_color:
            left_panel_bg_color = self.cget("background")
        style = ttk.Style(); style.configure("InfoArea.TFrame", background=left_panel_bg_color)

        right_content_frame = ttk.Frame(self)
        right_content_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=(5,10), pady=(10,10))
        right_content_frame.columnconfigure(0, weight=1)
        right_content_frame.rowconfigure(0, weight=3)
        right_content_frame.rowconfigure(1, weight=1)

        self.plot_display_area = PlotDisplayArea(right_content_frame, plot_size_pixels=300)
        self.plot_display_area.grid(row=0, column=0, sticky=tk.NSEW, pady=(0,5))

        self.info_display_area = InfoDisplayArea(right_content_frame, background_color=left_panel_bg_color)
        self.info_display_area.grid(row=1, column=0, sticky=tk.NSEW, pady=(5,0))

        # 3. Завершаем инициализацию SettingsPanel и делаем первый вызов update_profile_options
        if hasattr(self.settings_panel, 'update_profile_options'):
            self.settings_panel.update_profile_options(self.current_target_type)
        if hasattr(self.settings_panel, '_initial_ui_update'): # Этот вызов может быть избыточен, если update_profile_options все делает
            self.settings_panel._initial_ui_update() # Он вызовет update_profile_options еще раз

        # 4. Устанавливаем флаг, что ResultsWindow (и его компоненты) полностью инициализированы
        if hasattr(self.settings_panel, 'mark_initialization_complete'):
            self.settings_panel.mark_initialization_complete()

        # 5. Планируем первую отрисовку
        if PLOT_MODULE_AVAILABLE:
            self.after(100, self._recalculate_and_redraw_plots)
        else:
            messagebox.showerror("Ошибка импорта", "Модуль визуализации (plot.py) недоступен.", parent=self)
            if hasattr(self.info_display_area, 'update_uniformity_results'):
                self.info_display_area.update_uniformity_results("Ошибка plot.py")
        
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _recalculate_and_redraw_plots(self):
        print("--- ResultsWindow: _recalculate_and_redraw_plots (AttributeError Fix) ---")
        if not PLOT_MODULE_AVAILABLE:
            print("Отрисовка пропущена: PLOT_MODULE_AVAILABLE is False")
            return

        # Проверяем, что все необходимые атрибуты существуют перед их использованием
        if not hasattr(self, 'settings_panel') or \
           not hasattr(self, 'plot_display_area') or \
           not hasattr(self, 'info_display_area'):
            print("Warning: ResultsWindow не полностью инициализировано, пропуск _recalculate_and_redraw_plots.")
            return

        view_settings = self.settings_panel.get_view_settings()
        if view_settings is None:
            if hasattr(self.info_display_area, 'update_uniformity_results'): # Проверка перед использованием
                self.info_display_area.update_uniformity_results("Ошибка в параметрах вида")
            return

        self.current_vis_params['percent'] = view_settings['display_percent']
        self.current_vis_params['logscale'] = view_settings['use_logscale']
        smoothing_method = view_settings['smoothing']['method']
        smoothing_params_dict = view_settings['smoothing']['params']
        show_raw_profile_on_plot_effect = view_settings['show_raw_profile']
        profile_config_params = view_settings.get('profile_config_params', {})
        view_settings.setdefault('uniformity_method_key', self.settings_panel.get_uniformity_method())

        active_coverage_map = self.simulation_coverage_map_raw
        active_x_coords_edges = self.simulation_x_coords_edges
        active_y_coords_edges = self.simulation_y_coords_edges
        active_radius_grid_centers = self.simulation_radius_grid_centers
        active_target_params = self.simulation_target_params

        if active_coverage_map is None:
            if hasattr(self.info_display_area, 'update_uniformity_results'):
                self.info_display_area.update_uniformity_results("Нет данных")
            return

        profile_x_coords_centers = (active_x_coords_edges[:-1] + active_x_coords_edges[1:]) / 2.0 \
            if active_x_coords_edges is not None and len(active_x_coords_edges) > 1 else np.array([])
        profile_y_coords_centers = (active_y_coords_edges[:-1] + active_y_coords_edges[1:]) / 2.0 \
            if active_y_coords_edges is not None and len(active_y_coords_edges) > 1 else np.array([])

        profiles_for_stats_raw_list, \
        display_profile_coords_raw, \
        display_profile_values_raw, \
        display_profile_axis_label, \
        lines_to_draw_on_map = extract_profiles_for_statistics(
                active_coverage_map,
                profile_x_coords_centers,
                profile_y_coords_centers,
                active_x_coords_edges, 
                active_y_coords_edges, 
                self.current_target_type,
                profile_config_params
            )

        uniformity_result_text, first_prof_stats_dict = calculate_and_format_uniformity(
            profiles_for_stats_raw_list, view_settings, active_coverage_map
        )

        if hasattr(self.settings_panel, 'update_profile_stats'): # Проверка перед использованием
            if first_prof_stats_dict:
                self.settings_panel.update_profile_stats(
                    first_prof_stats_dict.get('t_max', '-'),
                    first_prof_stats_dict.get('t_min', '-'),
                    first_prof_stats_dict.get('t_mean', '-')
                )
            else:
                 self.settings_panel.update_profile_stats("-", "-", "-")

        if hasattr(self.info_display_area, 'update_uniformity_results'):
            self.info_display_area.update_uniformity_results(uniformity_result_text)

        processed_display_profile_values = None
        raw_comparison_profile_values = None
        raw_comparison_label = "Сырой (сравн.)"
        if display_profile_values_raw is not None and display_profile_coords_raw is not None and \
           len(display_profile_values_raw) > 0:
            base_profile_for_plot = display_profile_values_raw.copy()
            if self.current_vis_params['percent']:
                max_val_norm = np.nanmax(active_coverage_map.astype(float))
                if max_val_norm is not None and max_val_norm > 0:
                    base_profile_for_plot = base_profile_for_plot / max_val_norm * 100.0
                else: base_profile_for_plot = np.zeros_like(base_profile_for_plot)
            if show_raw_profile_on_plot_effect or smoothing_method == "Без сглаживания":
                processed_display_profile_values = base_profile_for_plot.copy()
                if smoothing_method != "Без сглаживания":
                    raw_comparison_profile_values = smooth_profile_data(
                        display_profile_coords_raw, base_profile_for_plot,
                        smoothing_method, smoothing_params_dict)
                    raw_comparison_label = "Сглаженный (сравн.)"
            else: 
                processed_display_profile_values = smooth_profile_data(
                    display_profile_coords_raw, base_profile_for_plot,
                    smoothing_method, smoothing_params_dict)
                raw_comparison_profile_values = base_profile_for_plot.copy()
                raw_comparison_label = "Сырой (сравн.)"
        
        map_fig = self.plot_display_area.get_map_figure()
        profile_fig = self.plot_display_area.get_profile_figure()
        if map_fig is None or profile_fig is None: return

        update_plots(
            map_figure=map_fig, profile_figure=profile_fig,
            coverage_map_data=active_coverage_map,
            x_coords_edges_data=active_x_coords_edges, y_coords_edges_data=active_y_coords_edges,
            radius_grid_centers_data=active_radius_grid_centers,
            target_params_data=active_target_params, current_vis_params_data=self.current_vis_params,
            display_profile_coords=display_profile_coords_raw,
            display_profile_values_processed=processed_display_profile_values,
            display_profile_axis_label=display_profile_axis_label,
            lines_to_draw_on_map=lines_to_draw_on_map,
            raw_profile_for_comparison_coords=display_profile_coords_raw,
            raw_profile_for_comparison_values=raw_comparison_profile_values,
            raw_profile_label=raw_comparison_label
        )
        self.plot_display_area.draw_canvases()
        
        if hasattr(self.settings_panel, 'enable_export_button'):
            self.settings_panel.enable_export_button() if active_coverage_map is not None else self.settings_panel.disable_export_button()
        print("Обновление графиков и статистики завершено (AttributeError Fix).")

    def _handle_export_excel(self): placeholder_export_excel(self)
    def _handle_calculate_mask(self):
        params = self.settings_panel.get_auto_uniformity_params() if hasattr(self.settings_panel, 'get_auto_uniformity_params') else None
        placeholder_calculate_mask(self, params)
    def _handle_load_profiles(self): placeholder_load_profiles(self)
    def _handle_reconstruct_map(self, reconstruction_method: str): placeholder_reconstruct_map(self, reconstruction_method)

    def _on_closing(self):
        try:
            if hasattr(self.plot_display_area, 'map_figure') and self.plot_display_area.map_figure:
                plt.close(self.plot_display_area.map_figure)
            if hasattr(self.plot_display_area, 'profile_figure') and self.plot_display_area.profile_figure:
                plt.close(self.plot_display_area.profile_figure)
        except Exception as e_close: print(f"Ошибка при закрытии фигур: {e_close}"); traceback.print_exc()
        finally: self.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Тест ResultsWindow (AttributeError Fix)")
    # ... (остальной тестовый код без изменений) ...
    num_edges_mock = config.SIM_GRID_SIZE if hasattr(config, 'SIM_GRID_SIZE') else 51
    num_cells_mock = num_edges_mock - 1
    mock_x_edges_data = np.linspace(-100, 100, num_edges_mock)
    mock_y_edges_data = np.linspace(-80, 80, num_edges_mock)
    mock_x_centers_data = (mock_x_edges_data[:-1] + mock_x_edges_data[1:]) / 2.0
    mock_y_centers_data = (mock_y_edges_data[:-1] + mock_y_edges_data[1:]) / 2.0
    mock_YC_centers, mock_XC_centers = np.meshgrid(mock_y_centers_data, mock_x_centers_data, indexing='ij')
    mock_radius_grid_centers_data = np.hypot(mock_XC_centers, mock_YC_centers)
    mock_coverage_map_data = 1000 * np.exp(-(mock_radius_grid_centers_data**2 / (2 * 50**2))) + np.random.rand(num_cells_mock, num_cells_mock) * 50
    mock_target_params_linear_data = {'target_type': config.TARGET_LINEAR, 'length': 200, 'width': 160, 'particles': 100000}
    mock_target_params_disk_data = {'target_type': config.TARGET_DISK, 'diameter': 150, 'particles': 100000}
    mock_vis_params_data = {'percent': True, 'logscale': False, 'show3d': False}
    def open_results_linear_test():
        try: ResultsWindow(root, mock_coverage_map_data, mock_x_edges_data, mock_y_edges_data, mock_radius_grid_centers_data, mock_target_params_linear_data, mock_vis_params_data)
        except Exception as e: print(f"Ошибка (Линейный): {e}"); traceback.print_exc()
    def open_results_disk_test():
        try: ResultsWindow(root, mock_coverage_map_data, mock_x_edges_data, mock_y_edges_data, mock_radius_grid_centers_data, mock_target_params_disk_data, mock_vis_params_data)
        except Exception as e: print(f"Ошибка (Диск): {e}"); traceback.print_exc()
    ttk.Button(root, text="Результаты (Линейный)", command=open_results_linear_test).pack(pady=10, padx=10, fill=tk.X)
    ttk.Button(root, text="Результаты (Диск)", command=open_results_disk_test).pack(pady=10, padx=10, fill=tk.X)
    root.mainloop()
