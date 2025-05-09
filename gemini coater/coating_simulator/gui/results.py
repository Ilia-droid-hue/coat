# coding: utf-8
# coating_simulator_project/coating_simulator/gui/results.py
"""
Toplevel window for displaying simulation results with interactive uniformity controls,
and a section for inverse problem solving (reconstructing map from profiles).
Layout v12.11 - Ensuring map plot fills its area without colorbar using direct axes positioning.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import traceback
import math
import os 

try:
    from ..visualization.plot import (plot_simulation_results as plot_on_figure,
                                      calculate_uniformity_stats,
                                      format_uniformity_stats,
                                      _smooth_profile as smooth_profile_data)
    PLOT_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"ОШИБКА ИМПОРТА из plot.py: {e}")
    PLOT_MODULE_AVAILABLE = False
    def plot_on_figure(*args, **kwargs): print("plot_on_figure (заглушка)")
    def calculate_uniformity_stats(*args, **kwargs): print("calculate_uniformity_stats (заглушка)"); return {}
    def format_uniformity_stats(*args, **kwargs): print("format_uniformity_stats (заглушка)"); return "Ошибка импорта plot.py"
    def smooth_profile_data(coords, data, method, params): print("smooth_profile_data (заглушка)"); return data

from .. import config

try:
    from .results_panels import SettingsPanel, PlotDisplayArea, InfoDisplayArea
except ImportError as e_panel:
    print(f"CRITICAL: Не удалось импортировать панели из .results_panels: {e_panel}")
    class SettingsPanel(ttk.Frame): #type: ignore
        def __init__(self, master, recalculate_callback, export_excel_callback, 
                     calculate_mask_callback, load_profiles_callback, reconstruct_map_callback, 
                     *args, **kwargs):
            super().__init__(master, *args, **kwargs)
            ttk.Label(self, text="SettingsPanel (ЗАГЛУШКА)").pack(fill=tk.BOTH, expand=True)
            self.recalculate_callback = recalculate_callback 
            self.export_excel_callback = export_excel_callback
            self.calculate_mask_callback = calculate_mask_callback
            self.load_profiles_callback = load_profiles_callback
            self.reconstruct_map_callback = reconstruct_map_callback
            ttk.Button(self, text="Обновить (заглушка)", command=self.recalculate_callback).pack()
        def get_uniformity_method(self): return 'U3'
        def update_profile_stats(self, *args): pass
        def get_view_settings(self): return {
            'smoothing': {'method': "Без сглаживания", 'params': {}},
            'display_percent': True, 'use_logscale': False, 'show_raw_profile': False}
        def get_auto_uniformity_params(self): return {'mode': 'Маска', 'mask_height': 0.0}
        def get_reconstruction_method(self): return "Линейная интерполяция"
        def update_loaded_files_text(self, text): pass
        def enable_export_button(self):pass
        def disable_export_button(self):pass
        def _initial_ui_update(self): pass

    class PlotDisplayArea(ttk.Frame): #type: ignore
        def __init__(self, master, plot_size_pixels=360, *args, **kwargs):
            super().__init__(master, *args, **kwargs)
            ttk.Label(self, text="PlotDisplayArea (ЗАГЛУШКА)").pack(fill=tk.BOTH, expand=True)
            self.map_figure = Figure() 
            self.profile_figure = Figure()
        def get_map_figure(self): return self.map_figure
        def get_profile_figure(self): return self.profile_figure
        def draw_canvases(self): pass

    class InfoDisplayArea(ttk.Frame): #type: ignore
        def __init__(self, master, background_color, *args, **kwargs):
            super().__init__(master, *args, **kwargs)
            ttk.Label(self, text="InfoDisplayArea (ЗАГЛУШКА)").pack(fill=tk.BOTH, expand=True)
        def update_uniformity_results(self, text): pass
        def update_auto_uniformity_info(self, text): pass
        def update_inverse_problem_info(self, text): pass

import matplotlib.pyplot as plt 
from matplotlib.figure import Figure 

class ResultsWindow(tk.Toplevel):
    """
    Окно для отображения результатов симуляции и анализа равномерности.
    Layout v12.11 - Ensuring map plot fills its area.
    """
    def __init__(self, parent, coverage_map: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, radius_grid: np.ndarray,
                 target_params: dict, vis_params: dict):
        super().__init__(parent)
        self.title("Результаты: Анализ, Сглаживание и Обратная Задача")
        self.geometry("1100x680") 
        self.minsize(850, 620)    

        self.simulation_coverage_map_raw = coverage_map
        self.simulation_x_coords = x_coords
        self.simulation_y_coords = y_coords
        self.simulation_radius_grid = radius_grid
        self.simulation_target_params = target_params
        self.current_vis_params = vis_params.copy() 

        self.loaded_profiles_data = [] 
        self.current_target_type = self.simulation_target_params.get('target_type', config.TARGET_DISK)

        self.columnconfigure(0, weight=0, minsize=270) 
        self.columnconfigure(1, weight=1)             
        self.rowconfigure(0, weight=1)                

        self.settings_panel = SettingsPanel(
            self, 
            recalculate_callback=self._placeholder_recalculate_and_redraw,
            export_excel_callback=self._placeholder_export_excel,
            calculate_mask_callback=self._placeholder_calculate_mask,
            load_profiles_callback=self._placeholder_load_profiles,  
            reconstruct_map_callback=self._placeholder_reconstruct_map 
        )
        self.settings_panel.grid(row=0, column=0, sticky=tk.NSEW, padx=(10,5), pady=(10,5))
            
        self.update_idletasks() 
        try: 
            s_temp = ttk.Style()
            left_panel_bg_color = s_temp.lookup("TFrame", "background")
        except tk.TclError:
            left_panel_bg_color = "SystemButtonFace" 
        
        if not isinstance(left_panel_bg_color, str) or not left_panel_bg_color : 
            left_panel_bg_color = self.cget("background")

        print(f"DEBUG: Determined left_panel_bg_color for info area: {left_panel_bg_color}")

        style = ttk.Style()
        style.configure("InfoArea.TFrame", background=left_panel_bg_color)
        
        right_content_frame = ttk.Frame(self) 
        right_content_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=(5,10), pady=(10,5)) 
        right_content_frame.columnconfigure(0, weight=1) 
        right_content_frame.rowconfigure(0, weight=0)    
        right_content_frame.rowconfigure(1, weight=0)    

        self.plot_display_area = PlotDisplayArea(right_content_frame, plot_size_pixels=350)
        self.plot_display_area.grid(row=0, column=0, sticky=tk.EW) 

        self.info_display_area = InfoDisplayArea(right_content_frame, background_color=left_panel_bg_color)
        self.info_display_area.grid(row=1, column=0, sticky=tk.EW + tk.S, pady=(5,0)) 

        if hasattr(self.settings_panel, '_initial_ui_update'): 
            self.settings_panel._initial_ui_update() 
        if PLOT_MODULE_AVAILABLE:
            self.after(200, self._placeholder_recalculate_and_redraw) 
        else:
            messagebox.showerror("Ошибка импорта", "Не удалось загрузить модуль визуализации (plot.py).\nГрафики не будут отображены.", parent=self)
            if hasattr(self.info_display_area, 'update_uniformity_results'): 
                self.info_display_area.update_uniformity_results("Ошибка импорта plot.py")

    def _placeholder_recalculate_and_redraw(self):
        print("--- ResultsWindow: Пересчет и перерисовка ---")
        if not PLOT_MODULE_AVAILABLE:
            messagebox.showerror("Ошибка", "Модуль построения графиков недоступен.", parent=self)
            return

        view_settings = self.settings_panel.get_view_settings()
        if view_settings is None: 
            if hasattr(self.info_display_area, 'update_uniformity_results'):
                self.info_display_area.update_uniformity_results("Ошибка в параметрах вида/сглаживания")
            return
            
        self.current_vis_params['percent'] = view_settings['display_percent']
        self.current_vis_params['logscale'] = view_settings['use_logscale']
        
        smoothing_method = view_settings['smoothing']['method']
        smoothing_params_dict = view_settings['smoothing']['params']

        active_coverage_map = self.simulation_coverage_map_raw
        active_x_coords = self.simulation_x_coords
        active_y_coords = self.simulation_y_coords
        active_radius_grid = self.simulation_radius_grid
        active_target_params = self.simulation_target_params

        if active_coverage_map is None:
            messagebox.showwarning("Нет данных", "Данные симуляции отсутствуют для отображения.", parent=self)
            return
        
        raw_profile_coords_1d = None
        raw_profile_values_1d = None
        profile_axis_label = "Позиция (мм)"
        target_type = active_target_params.get('target_type', config.TARGET_DISK)

        if target_type == config.TARGET_LINEAR:
            if active_coverage_map.ndim == 2 and active_x_coords is not None and active_coverage_map.shape[1] == len(active_x_coords):
                raw_profile_values_1d = np.nanmean(active_coverage_map, axis=0)
                raw_profile_coords_1d = active_x_coords
                profile_axis_label = "Позиция X (мм)"
            else:
                raw_profile_coords_1d = active_x_coords if active_x_coords is not None else np.array([])
                raw_profile_values_1d = np.zeros_like(raw_profile_coords_1d)
        elif target_type in [config.TARGET_DISK, config.TARGET_DOME, config.TARGET_PLANETARY]:
            if active_coverage_map.ndim == 2 and active_x_coords is not None:
                center_y_idx = active_coverage_map.shape[0] // 2
                raw_profile_values_1d = active_coverage_map[center_y_idx, :]
                raw_profile_coords_1d = active_x_coords
                profile_axis_label = "Позиция X (мм)"
            else:
                raw_profile_coords_1d = active_x_coords if active_x_coords is not None else np.array([])
                raw_profile_values_1d = np.zeros_like(raw_profile_coords_1d)
        else:
            raw_profile_coords_1d = active_x_coords if active_x_coords is not None else np.array([])
            raw_profile_values_1d = np.zeros_like(raw_profile_coords_1d)

        if raw_profile_coords_1d is None or raw_profile_values_1d is None or len(raw_profile_coords_1d) == 0:
             messagebox.showerror("Ошибка", "Не удалось извлечь данные для 1D профиля.", parent=self)
             return

        profile_data_for_stats = raw_profile_values_1d.copy()
        profile_data_for_plot = raw_profile_values_1d.copy()

        if self.current_vis_params['percent']:
            max_val_for_norm = np.nanmax(active_coverage_map.astype(float))
            if max_val_for_norm <=0: max_val_for_norm = 1.0
            
            profile_data_for_stats = profile_data_for_stats / max_val_for_norm * 100.0
            profile_data_for_plot = profile_data_for_plot / max_val_for_norm * 100.0

        smoothed_profile_for_stats = smooth_profile_data(
            raw_profile_coords_1d, profile_data_for_stats,
            smoothing_method, smoothing_params_dict
        )
        
        stats_source_data = smoothed_profile_for_stats
        if view_settings['show_raw_profile'] or smoothing_method == "Без сглаживания":
            stats_source_data = profile_data_for_stats

        stats = calculate_uniformity_stats(stats_source_data)
        
        self.settings_panel.update_profile_stats(
            f"{np.nanmax(stats_source_data):.2f}" if np.any(np.isfinite(stats_source_data)) else "-",
            f"{np.nanmin(stats_source_data):.2f}" if np.any(np.isfinite(stats_source_data)) else "-",
            f"{np.nanmean(stats_source_data):.2f}" if np.any(np.isfinite(stats_source_data)) else "-"
        )

        selected_method_U = self.settings_panel.get_uniformity_method()
        uniformity_result_text = format_uniformity_stats(stats, selected_method_U)
        self.info_display_area.update_uniformity_results(uniformity_result_text)

        map_fig = self.plot_display_area.get_map_figure()
        profile_fig = self.plot_display_area.get_profile_figure()

        if map_fig is None or profile_fig is None:
            print("Ошибка: Фигуры для графиков не инициализированы в PlotDisplayArea.")
            return

        map_fig.clear()
        profile_fig.clear()
        
        plot_on_figure(
            fig=map_fig, coverage_map=active_coverage_map,
            x_coords=active_x_coords, y_coords=active_y_coords, radius_grid=active_radius_grid,
            target_params=active_target_params, vis_params=self.current_vis_params,
            profile_1d_coords=raw_profile_coords_1d, 
            profile_1d_values=profile_data_for_plot,
            show_colorbar=False 
        )
        if len(map_fig.axes) > 1: 
            map_fig.delaxes(map_fig.axes[1]) 
        if len(map_fig.axes) > 0:
             ax_map = map_fig.axes[0]
             ax_map.set_title('Карта покрытия', fontsize=10)
             # Устанавливаем позицию осей карты, чтобы они занимали почти всю фигуру
             # [left, bottom, width, height] в долях от размера фигуры
             ax_map.set_position([0.01, 0.01, 0.98, 0.92]) # Оставляем немного места сверху для заголовка
             # Восстанавливаем соотношение сторон, если оно было 'equal'
             if 'equal' in str(ax_map.get_aspect()).lower(): # Проверяем, было ли установлено 'equal'
                 ax_map.set_aspect('equal', adjustable='box') 
             else:
                 ax_map.set_aspect('auto')


        ax_profile_actual = profile_fig.add_subplot(111)
        
        profile_to_display_on_plot = profile_data_for_plot
        if not view_settings['show_raw_profile'] and smoothing_method != "Без сглаживания":
            profile_to_display_on_plot = smooth_profile_data(raw_profile_coords_1d, profile_data_for_plot, 
                                                             smoothing_method, smoothing_params_dict)

        ax_profile_actual.plot(raw_profile_coords_1d, profile_to_display_on_plot, '-', color='orange', linewidth=1.5, label='Профиль')

        if view_settings['show_raw_profile'] and smoothing_method != "Без сглаживания":
            smoothed_for_comparison = smooth_profile_data(raw_profile_coords_1d, profile_data_for_plot, smoothing_method, smoothing_params_dict)
            if not np.allclose(profile_data_for_plot, smoothed_for_comparison, equal_nan=True):
                ax_profile_actual.plot(raw_profile_coords_1d, smoothed_for_comparison, '--', color='blue', linewidth=1.0, label='Сглаженный')
        elif not view_settings['show_raw_profile'] and smoothing_method != "Без сглаживания":
             if not np.allclose(profile_data_for_plot, profile_to_display_on_plot, equal_nan=True): 
                ax_profile_actual.plot(raw_profile_coords_1d, profile_data_for_plot, ':', color='gray', linewidth=1.0, label='Сырой')

        ax_profile_actual.set_title('Профиль покрытия', fontsize=10)
        ax_profile_actual.set_xlabel(profile_axis_label, fontsize=9)
        ax_profile_actual.set_ylabel('Покрытие (%)' if self.current_vis_params['percent'] else 'Количество частиц', fontsize=9)
        ax_profile_actual.grid(True, linestyle=':')
        if ax_profile_actual.has_data(): 
            ax_profile_actual.legend(fontsize='small')

        if np.any(np.isfinite(profile_to_display_on_plot)):
            y_min_plot = np.nanmin(profile_to_display_on_plot)
            y_max_plot = np.nanmax(profile_to_display_on_plot)
            padding = (y_max_plot - y_min_plot) * 0.05 if (y_max_plot - y_min_plot) > 1e-6 else 0.1
            final_y_min = y_min_plot - padding
            final_y_max = y_max_plot + padding
            if self.current_vis_params['percent']:
                final_y_min = max(0, final_y_min)
                final_y_max = min(110, final_y_max) if final_y_max > 0 else 10
                if final_y_max <= final_y_min : final_y_max = final_y_min + 10
            ax_profile_actual.set_ylim(bottom=final_y_min, top=final_y_max)
        else:
            ax_profile_actual.set_ylim(0, 1 if not self.current_vis_params['percent'] else 10)

        # tight_layout для map_fig больше не нужен, так как мы используем set_position
        profile_fig.tight_layout(pad=0.8) 

        self.plot_display_area.draw_canvases()
        
        self.settings_panel.enable_export_button() if active_coverage_map is not None else self.settings_panel.disable_export_button()
        print("Обновление графиков завершено.")

    def _placeholder_calculate_mask(self):
         params = self.settings_panel.get_auto_uniformity_params()
         if params:
            messagebox.showinfo("Заглушка", f"Расчет маски для режима '{params['mode']}' с высотой {params['mask_height']}мм еще не реализован.", parent=self)
            self.info_display_area.update_auto_uniformity_info(f"Расчет маски ({params['mode']})... (заглушка)")

    def _placeholder_export_excel(self):
         messagebox.showinfo("Заглушка", "Функция экспорта в Excel еще не реализована.", parent=self)

    def _placeholder_load_profiles(self):
        filepaths = filedialog.askopenfilenames(
            title="Выберите файлы профилей (.csv, .txt)",
            filetypes=[("CSV файлы", "*.csv"), ("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")]
        )
        if filepaths:
            self.loaded_profiles_data = [] 
            filenames = [os.path.basename(fp) for fp in filepaths]
            self.settings_panel.update_loaded_files_text("Загружено: " + ", ".join(filenames) if filenames else "Файлы не загружены")
            self.info_display_area.update_inverse_problem_info(f"Загружено файлов: {len(filepaths)}.\nГотово к реконструкции.")
            messagebox.showinfo("Загрузка", f"Выбрано файлов: {len(filepaths)}.\nДальнейшая обработка и реконструкция еще не реализованы.", parent=self)
        else:
            self.settings_panel.update_loaded_files_text("Загрузка отменена")
            self.info_display_area.update_inverse_problem_info("Файлы профилей не загружены.")

    def _placeholder_reconstruct_map(self, reconstruction_method: str): 
        if not self.settings_panel.loaded_files_text_var.get().startswith("Загружено:"): 
            messagebox.showwarning("Обратная задача", "Сначала загрузите файлы профилей.", parent=self)
            self.info_display_area.update_inverse_problem_info("Ошибка: Профили не загружены.")
            return
        
        self.info_display_area.update_inverse_problem_info(f"Реконструкция ({reconstruction_method})...\n(Функция не реализована)")
        messagebox.showinfo("Заглушка", f"Функция построения карты по профилям ({reconstruction_method}) еще не реализована.", parent=self)


# --- Example Usage ---
if __name__ == '__main__':
    root = tk.Tk()
    root.title("Тест ResultsWindow GUI v12.11") 
    root.geometry("200x100")

    grid_res = 50
    mock_x = np.linspace(-100, 100, grid_res)
    mock_y = np.linspace(-80, 80, grid_res)
    mock_X, mock_Y = np.meshgrid(mock_x, mock_y)
    mock_rr = np.hypot(mock_X, mock_Y)
    sigma_map = 50
    mock_coverage = 1000 * np.exp(-(mock_rr**2 / (2 * sigma_map**2)))
    mock_coverage += np.random.rand(grid_res, grid_res) * 50

    mock_target_params = {'target_type': config.TARGET_DISK, 'diameter': 200}
    mock_vis_params = {'percent': True, 'logscale': False}

    def open_results():
        print("Открытие окна результатов (GUI v12.11)...")
        try:
            ResultsWindow(root, mock_coverage, mock_x, mock_y, mock_rr, mock_target_params, mock_vis_params)
        except Exception as e:
            print(f"Ошибка при открытии ResultsWindow: {e}")
            traceback.print_exc()

    button = ttk.Button(root, text="Показать окно (GUI v12.11)", command=open_results)
    button.pack(pady=20)
    root.mainloop()
