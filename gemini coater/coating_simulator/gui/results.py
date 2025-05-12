# coding: utf-8
# coating_simulator_project/coating_simulator/gui/results.py
"""
Toplevel window for displaying simulation results with interactive uniformity controls.
Calculates and displays source efficiency.
Ensures profile and uniformity are calculated correctly.
Allows selecting X or Y profile for linear targets, and multiple axes for circular.
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
        def __init__(self, master, recalculate_callback, export_excel_callback, calculate_mask_callback, load_profiles_callback, reconstruct_map_callback, *args, **kwargs): super().__init__(master, *args, **kwargs); ttk.Label(self, text="SettingsPanel (ЗАГЛУШКА)").pack(fill=tk.BOTH, expand=True); self.recalculate_callback = recalculate_callback; self.export_excel_callback = export_excel_callback; self.calculate_mask_callback = calculate_mask_callback; self.load_profiles_callback = load_profiles_callback; self.reconstruct_map_callback = reconstruct_map_callback; ttk.Button(self, text="Обновить (заглушка)", command=self.recalculate_callback).pack()
        def get_uniformity_method(self): return 'U3'
        def update_profile_stats(self, *args): pass
        def get_view_settings(self): return {'smoothing': {'method': "Без сглаживания", 'params': {}},'display_percent': True, 'use_logscale': False, 'show_raw_profile': False, 'uniformity_profile_config': 'X'} 
        def get_auto_uniformity_params(self): return {'mode': 'Маска', 'mask_height': 0.0}
        def get_reconstruction_method(self): return "Линейная интерполяция"
        def update_loaded_files_text(self, text): pass
        def enable_export_button(self):pass
        def disable_export_button(self):pass
        def _initial_ui_update(self): pass
        def update_profile_options(self, target_type: str | None): print(f"SettingsPanel: update_profile_options({target_type}) (заглушка)")
    class PlotDisplayArea(ttk.Frame): #type: ignore
        def __init__(self, master, plot_size_pixels=360, *args, **kwargs): super().__init__(master, *args, **kwargs); ttk.Label(self, text="PlotDisplayArea (ЗАГЛУШКА)").pack(fill=tk.BOTH, expand=True); from matplotlib.figure import Figure; self.map_figure = Figure(); self.profile_figure = Figure()
        def get_map_figure(self): return self.map_figure
        def get_profile_figure(self): return self.profile_figure
        def draw_canvases(self): pass
    class InfoDisplayArea(ttk.Frame): #type: ignore
        def __init__(self, master, background_color, *args, **kwargs): super().__init__(master, *args, **kwargs); ttk.Label(self, text="InfoDisplayArea (ЗАГЛУШКА)").pack(fill=tk.BOTH, expand=True)
        def update_uniformity_results(self, text): pass
        def update_auto_uniformity_info(self, text): pass
        def update_inverse_problem_info(self, text): pass

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

class ResultsWindow(tk.Toplevel):
    def __init__(self, parent, coverage_map: np.ndarray, 
                 x_coords_edges: np.ndarray, y_coords_edges: np.ndarray, 
                 radius_grid_centers: np.ndarray,
                 target_params: dict, vis_params: dict):
        super().__init__(parent)
        self.title("Результаты: Анализ и Профили")
        self.geometry("1100x680") 
        self.minsize(850, 620)

        self.simulation_coverage_map_raw = coverage_map
        self.simulation_x_coords_edges = x_coords_edges
        self.simulation_y_coords_edges = y_coords_edges
        self.simulation_radius_grid_centers = radius_grid_centers
        self.simulation_target_params = target_params
        self.current_vis_params = vis_params.copy()

        self.loaded_profiles_data = []
        self.current_target_type = self.simulation_target_params.get('target_type', config.TARGET_DISK)

        self.columnconfigure(0, weight=0, minsize=270)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.settings_panel = SettingsPanel(self,recalculate_callback=self._placeholder_recalculate_and_redraw,export_excel_callback=self._placeholder_export_excel,calculate_mask_callback=self._placeholder_calculate_mask,load_profiles_callback=self._placeholder_load_profiles,reconstruct_map_callback=self._placeholder_reconstruct_map)
        self.settings_panel.grid(row=0, column=0, sticky=tk.NSEW, padx=(10,5), pady=(10,5))

        if hasattr(self.settings_panel, 'update_profile_options'):
            self.settings_panel.update_profile_options(self.current_target_type)

        self.update_idletasks()
        try: s_temp = ttk.Style(); left_panel_bg_color = s_temp.lookup("TFrame", "background")
        except tk.TclError: left_panel_bg_color = "SystemButtonFace"
        if not isinstance(left_panel_bg_color, str) or not left_panel_bg_color : left_panel_bg_color = self.cget("background")
        style = ttk.Style(); style.configure("InfoArea.TFrame", background=left_panel_bg_color)

        right_content_frame = ttk.Frame(self)
        right_content_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=(5,10), pady=(10,5))
        right_content_frame.columnconfigure(0, weight=1)
        right_content_frame.rowconfigure(0, weight=0) 
        right_content_frame.rowconfigure(1, weight=0) 

        self.plot_display_area = PlotDisplayArea(right_content_frame, plot_size_pixels=350)
        self.plot_display_area.grid(row=0, column=0, sticky=tk.EW)
        self.info_display_area = InfoDisplayArea(right_content_frame, background_color=left_panel_bg_color)
        self.info_display_area.grid(row=1, column=0, sticky=tk.EW + tk.S, pady=(5,0))

        if hasattr(self.settings_panel, '_initial_ui_update'): self.settings_panel._initial_ui_update()
        if PLOT_MODULE_AVAILABLE: self.after(100, self._placeholder_recalculate_and_redraw)
        else:
            messagebox.showerror("Ошибка импорта", "Не удалось загрузить модуль визуализации (plot.py).\nГрафики не будут отображены.", parent=self)
            if hasattr(self.info_display_area, 'update_uniformity_results'): self.info_display_area.update_uniformity_results("Ошибка импорта plot.py")
    
    def _get_diagonal_profile(self, coverage_map, main_diagonal=True):
        if coverage_map is None or coverage_map.ndim != 2:
            return np.array([]), np.array([])
        rows, cols = coverage_map.shape
        diag_len = min(rows, cols)
        if diag_len == 0: return np.array([]), np.array([])
        
        if main_diagonal: # Главная диагональ (от [0,0])
            profile = np.array([coverage_map[i, i] for i in range(diag_len)])
        else: # Антидиагональ (от [0, cols-1])
            profile = np.array([coverage_map[i, cols - 1 - i] for i in range(diag_len)])
        
        # Для диагоналей координаты могут быть просто индексами или расстоянием от угла/центра
        # Для простоты расчета равномерности, достаточно последовательности значений.
        # Для отображения на графике, если потребуется, нужно будет рассчитать реальные координаты.
        # Пока используем простые индексы для coords.
        coords = np.arange(diag_len) 
        return coords, profile


    def _placeholder_recalculate_and_redraw(self):
        print("--- ResultsWindow: Пересчет и перерисовка (динамические профили) ---")
        if not PLOT_MODULE_AVAILABLE: return

        view_settings = self.settings_panel.get_view_settings()
        if view_settings is None:
            if hasattr(self.info_display_area, 'update_uniformity_results'): self.info_display_area.update_uniformity_results("Ошибка в параметрах вида")
            return

        self.current_vis_params['percent'] = view_settings['display_percent']
        self.current_vis_params['logscale'] = view_settings['use_logscale']
        smoothing_method = view_settings['smoothing']['method']
        smoothing_params_dict = view_settings['smoothing']['params']
        
        uniformity_profile_config = view_settings.get('uniformity_profile_config', 'X') 

        active_coverage_map = self.simulation_coverage_map_raw
        active_x_coords_edges = self.simulation_x_coords_edges
        active_y_coords_edges = self.simulation_y_coords_edges
        active_radius_grid_centers = self.simulation_radius_grid_centers
        active_target_params = self.simulation_target_params

        if active_coverage_map is None:
            messagebox.showwarning("Нет данных", "Данные симуляции отсутствуют для отображения.", parent=self)
            if hasattr(self.info_display_area, 'update_uniformity_results'): self.info_display_area.update_uniformity_results("Нет данных")
            return

        profile_x_coords_centers = np.array([])
        if active_x_coords_edges is not None and len(active_x_coords_edges) > 1:
            profile_x_coords_centers = (active_x_coords_edges[:-1] + active_x_coords_edges[1:]) / 2.0
        
        profile_y_coords_centers = np.array([])
        if active_y_coords_edges is not None and len(active_y_coords_edges) > 1:
            profile_y_coords_centers = (active_y_coords_edges[:-1] + active_y_coords_edges[1:]) / 2.0

        profiles_for_stats = []
        display_profile_coords = None
        display_profile_values = None
        display_profile_axis_label = "Позиция (мм)"
        target_type = active_target_params.get('target_type', config.TARGET_DISK)

        # --- Логика извлечения профилей ---
        if active_coverage_map.ndim == 2: # Убедимся, что карта 2D
            num_rows, num_cols = active_coverage_map.shape

            if target_type == config.TARGET_LINEAR:
                if uniformity_profile_config == 'Y':
                    if num_rows > 0 and len(profile_y_coords_centers) == num_rows:
                        profile_data = np.nanmean(active_coverage_map, axis=1) # Усреднение по X
                        profiles_for_stats.append({'coords': profile_y_coords_centers, 'values': profile_data, 'label': 'Y'})
                        display_profile_coords = profile_y_coords_centers
                        display_profile_values = profile_data
                        display_profile_axis_label = "Позиция Y (мм)"
                else: # 'X' или по умолчанию
                    if num_cols > 0 and len(profile_x_coords_centers) == num_cols:
                        profile_data = np.nanmean(active_coverage_map, axis=0) # Усреднение по Y
                        profiles_for_stats.append({'coords': profile_x_coords_centers, 'values': profile_data, 'label': 'X'})
                        display_profile_coords = profile_x_coords_centers
                        display_profile_values = profile_data
                        display_profile_axis_label = "Позиция X (мм)"
            
            elif target_type in [config.TARGET_DISK, config.TARGET_DOME, config.TARGET_PLANETARY]:
                center_y_idx = num_rows // 2
                center_x_idx = num_cols // 2

                if uniformity_profile_config in ['1H', '2HV', '4HVD']: # Горизонтальный
                    if num_cols > 0 and len(profile_x_coords_centers) == num_cols:
                        profile_h_data = active_coverage_map[center_y_idx, :]
                        profiles_for_stats.append({'coords': profile_x_coords_centers, 'values': profile_h_data, 'label': 'X (гориз.)'})
                        if display_profile_values is None:
                            display_profile_coords = profile_x_coords_centers
                            display_profile_values = profile_h_data
                            display_profile_axis_label = "Позиция X (мм)"
                
                if uniformity_profile_config in ['2HV', '4HVD']: # Вертикальный
                    if num_rows > 0 and len(profile_y_coords_centers) == num_rows:
                        profile_v_data = active_coverage_map[:, center_x_idx]
                        profiles_for_stats.append({'coords': profile_y_coords_centers, 'values': profile_v_data, 'label': 'Y (верт.)'})
                        if display_profile_values is None: # Отображаем, если горизонтальный не был выбран/извлечен
                            display_profile_coords = profile_y_coords_centers
                            display_profile_values = profile_v_data
                            display_profile_axis_label = "Позиция Y (мм)"

                if uniformity_profile_config == '4HVD': # Диагональные
                    diag_coords1, diag_values1 = self._get_diagonal_profile(active_coverage_map, main_diagonal=True)
                    if diag_values1.size > 1 : profiles_for_stats.append({'coords': diag_coords1, 'values': diag_values1, 'label': 'Диаг. 1'})
                    
                    diag_coords2, diag_values2 = self._get_diagonal_profile(active_coverage_map, main_diagonal=False)
                    if diag_values2.size > 1 : profiles_for_stats.append({'coords': diag_coords2, 'values': diag_values2, 'label': 'Диаг. 2'})
            else: # Другие типы или ошибка - по умолчанию X-профиль
                 if num_cols > 0 and len(profile_x_coords_centers) == num_cols:
                    center_y_idx_map = num_rows // 2
                    profile_data = active_coverage_map[center_y_idx_map, :]
                    profiles_for_stats.append({'coords': profile_x_coords_centers, 'values': profile_data, 'label': 'X'})
                    display_profile_coords = profile_x_coords_centers
                    display_profile_values = profile_data
                    display_profile_axis_label = "Позиция X (мм)"
        
        # Если display_profile_values все еще None, значит профили не извлеклись
        if display_profile_values is None or display_profile_coords is None or len(display_profile_values) == 0:
            print("Предупреждение: Не удалось извлечь основной 1D профиль для отображения.")
            display_profile_coords = np.array([0.0]) if display_profile_coords is None or len(display_profile_coords) == 0 else display_profile_coords
            display_profile_values = np.array([0.0] * len(display_profile_coords)) if display_profile_coords.size > 0 else np.array([0.0])
            uniformity_result_text = "U: Нет данных для профиля"
            self.settings_panel.update_profile_stats("-", "-", "-")
        else:
            display_profile_values_for_stats = display_profile_values.copy()
            display_profile_values_for_plot = display_profile_values.copy()

            if self.current_vis_params['percent']:
                max_val_for_norm_map = np.nanmax(active_coverage_map.astype(float))
                if max_val_for_norm_map <=0: max_val_for_norm_map = 1.0
                # Нормируем все профили, которые пойдут в статистику
                for prof_dict in profiles_for_stats:
                    prof_dict['values_norm'] = prof_dict['values'] / max_val_for_norm_map * 100.0
                # Нормируем отображаемый профиль
                display_profile_values_for_plot = display_profile_values / max_val_for_norm_map * 100.0
            else: # Если не в %, то создаем ключ 'values_norm' равный 'values'
                for prof_dict in profiles_for_stats:
                    prof_dict['values_norm'] = prof_dict['values'].copy()
            
            uniformity_texts = []
            first_profile_stats_updated = False
            if profiles_for_stats:
                for prof_data_dict in profiles_for_stats:
                    coords = prof_data_dict['coords']
                    # Для статистики используем values_norm (уже нормированные, если нужно)
                    values_current_for_stats = prof_data_dict.get('values_norm', prof_data_dict['values']) 

                    if values_current_for_stats is None or len(coords) != len(values_current_for_stats) or len(values_current_for_stats) < 2:
                        uniformity_texts.append(f"U ({prof_data_dict['label']}): Нет данных")
                        continue
                    
                    if np.any(np.isfinite(values_current_for_stats)):
                        smoothed_current = smooth_profile_data(coords, values_current_for_stats, smoothing_method, smoothing_params_dict)
                        stats_source_current = smoothed_current
                        if view_settings['show_raw_profile'] or smoothing_method == "Без сглаживания":
                            stats_source_current = values_current_for_stats
                        
                        stats_current = calculate_uniformity_stats(stats_source_current)
                        selected_method_U = self.settings_panel.get_uniformity_method()
                        u_text = format_uniformity_stats(stats_current, selected_method_U).replace(selected_method_U, f"{selected_method_U} ({prof_data_dict['label']})")
                        uniformity_texts.append(u_text)

                        # Обновляем Max/Min/Mean только для первого профиля в списке profiles_for_stats
                        # (или для того, который выбран для отображения, если это реализовать)
                        if not first_profile_stats_updated:
                             self.settings_panel.update_profile_stats(
                                f"{np.nanmax(stats_source_current):.2f}" if np.any(np.isfinite(stats_source_current)) else "-",
                                f"{np.nanmin(stats_source_current):.2f}" if np.any(np.isfinite(stats_source_current)) else "-",
                                f"{np.nanmean(stats_source_current):.2f}" if np.any(np.isfinite(stats_source_current)) else "-"
                            )
                             first_profile_stats_updated = True
                    else:
                        uniformity_texts.append(f"U ({prof_data_dict['label']}): Нет валидных данных")
                uniformity_result_text = "\n".join(uniformity_texts) if uniformity_texts else "U: Нет данных"
            else: 
                self.settings_panel.update_profile_stats("-", "-", "-")
                uniformity_result_text = "U: Нет профилей для расчета"

        if hasattr(self.info_display_area, 'update_uniformity_results'):
            self.info_display_area.update_uniformity_results(uniformity_result_text)

        map_fig = self.plot_display_area.get_map_figure(); profile_fig = self.plot_display_area.get_profile_figure()
        if map_fig is None or profile_fig is None: return
        map_fig.clear(); profile_fig.clear()

        plot_on_figure(fig=map_fig, coverage_map=active_coverage_map, x_coords=active_x_coords_edges, y_coords=active_y_coords_edges, radius_grid=active_radius_grid_centers, target_params=active_target_params, vis_params=self.current_vis_params, show_colorbar=True, plot_type="map_only")
        map_fig.tight_layout(pad=1.0)

        ax_profile_actual = profile_fig.add_subplot(111)
        
        # Используем display_profile_values_for_plot и display_profile_coords для графика
        profile_to_display_on_plot_final = display_profile_values_for_plot # Уже нормирован, если надо

        if np.any(np.isfinite(profile_to_display_on_plot_final)) and len(profile_to_display_on_plot_final) > 1:
            if not view_settings['show_raw_profile'] and smoothing_method != "Без сглаживания":
                profile_to_display_on_plot_final = smooth_profile_data(display_profile_coords, profile_to_display_on_plot_final, smoothing_method, smoothing_params_dict)
            
            ax_profile_actual.plot(display_profile_coords, profile_to_display_on_plot_final, '-', color='orange', linewidth=1.5, label='Профиль')
            
            # Отображение "сырого" или "сглаженного" для сравнения
            raw_profile_for_comparison = display_profile_values.copy() # Берем исходный display_profile_values
            if self.current_vis_params['percent']: # Нормируем его, если нужно
                max_val_for_norm_map = np.nanmax(active_coverage_map.astype(float))
                if max_val_for_norm_map <=0: max_val_for_norm_map = 1.0
                raw_profile_for_comparison = raw_profile_for_comparison / max_val_for_norm_map * 100.0

            if view_settings['show_raw_profile'] and smoothing_method != "Без сглаживания":
                # Если показываем сырой, а на графике сглаженный, то сглаженный уже profile_to_display_on_plot_final
                # Нужно убедиться, что profile_to_display_on_plot_final действительно сглаженный
                is_smoothed_on_plot = not np.allclose(raw_profile_for_comparison, profile_to_display_on_plot_final, equal_nan=True)
                if is_smoothed_on_plot: # Если на графике сглаженный, а мы хотим показать сырой
                    ax_profile_actual.plot(display_profile_coords, raw_profile_for_comparison, ':', color='gray', linewidth=1.0, label='Сырой (для сравнения)')

            elif not view_settings['show_raw_profile'] and smoothing_method != "Без сглаживания":
                 # Если показываем сглаженный, а хотим добавить сырой для сравнения (если они отличаются)
                 if not np.allclose(raw_profile_for_comparison, profile_to_display_on_plot_final, equal_nan=True):
                    ax_profile_actual.plot(display_profile_coords, raw_profile_for_comparison, ':', color='gray', linewidth=1.0, label='Сырой')
        
        ax_profile_actual.set_title('Профиль покрытия', fontsize=10)
        ax_profile_actual.set_xlabel(display_profile_axis_label, fontsize=9)
        ax_profile_actual.set_ylabel('Покрытие (%)' if self.current_vis_params['percent'] else 'Количество частиц', fontsize=9)
        ax_profile_actual.grid(True, linestyle=':')
        if ax_profile_actual.has_data(): ax_profile_actual.legend(fontsize='small')

        if np.any(np.isfinite(profile_to_display_on_plot_final)):
            y_min_plot = np.nanmin(profile_to_display_on_plot_final); y_max_plot = np.nanmax(profile_to_display_on_plot_final)
            padding = (y_max_plot - y_min_plot) * 0.05 if (y_max_plot - y_min_plot) > 1e-6 else 0.1
            final_y_min = y_min_plot - padding; final_y_max = y_max_plot + padding
            if self.current_vis_params['percent']:
                final_y_min = max(0, final_y_min)
                final_y_max = min(110, final_y_max) if final_y_max > 0 else 10
                if final_y_max <= final_y_min : final_y_max = final_y_min + 10
            ax_profile_actual.set_ylim(bottom=final_y_min, top=final_y_max)
        else:
            ax_profile_actual.set_ylim(0, 1 if not self.current_vis_params['percent'] else 10)
            ax_profile_actual.text(0.5, 0.5, "Нет данных для профиля", horizontalalignment='center', verticalalignment='center', transform=ax_profile_actual.transAxes)

        profile_fig.tight_layout(pad=0.8)
        self.plot_display_area.draw_canvases()
        self.settings_panel.enable_export_button() if active_coverage_map is not None else self.settings_panel.disable_export_button()
        print("Обновление графиков завершено.")

    def _placeholder_calculate_mask(self):
         params = self.settings_panel.get_auto_uniformity_params()
         if params: messagebox.showinfo("Заглушка", f"Расчет маски для режима '{params['mode']}' с высотой {params['mask_height']}мм еще не реализован.", parent=self); self.info_display_area.update_auto_uniformity_info(f"Расчет маски ({params['mode']})... (заглушка)")
    def _placeholder_export_excel(self): messagebox.showinfo("Заглушка", "Функция экспорта в Excel еще не реализована.", parent=self)
    def _placeholder_load_profiles(self):
        filepaths = filedialog.askopenfilenames(title="Выберите файлы профилей (.csv, .txt)", filetypes=[("CSV файлы", "*.csv"), ("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")])
        if filepaths: self.loaded_profiles_data = []; filenames = [os.path.basename(fp) for fp in filepaths]; self.settings_panel.update_loaded_files_text("Загружено: " + ", ".join(filenames) if filenames else "Файлы не загружены"); self.info_display_area.update_inverse_problem_info(f"Загружено файлов: {len(filepaths)}.\nГотово к реконструкции."); messagebox.showinfo("Загрузка", f"Выбрано файлов: {len(filepaths)}.\nДальнейшая обработка и реконструкция еще не реализованы.", parent=self)
        else: self.settings_panel.update_loaded_files_text("Загрузка отменена"); self.info_display_area.update_inverse_problem_info("Файлы профилей не загружены.")
    def _placeholder_reconstruct_map(self, reconstruction_method: str):
        if not self.settings_panel.loaded_files_text_var.get().startswith("Загружено:"): messagebox.showwarning("Обратная задача", "Сначала загрузите файлы профилей.", parent=self); self.info_display_area.update_inverse_problem_info("Ошибка: Профили не загружены."); return
        self.info_display_area.update_inverse_problem_info(f"Реконструкция ({reconstruction_method})...\n(Функция не реализована)"); messagebox.showinfo("Заглушка", f"Функция построения карты по профилям ({reconstruction_method}) еще не реализована.", parent=self)


if __name__ == '__main__':
    root = tk.Tk()
    root.title("Тест ResultsWindow GUI v12.16 (Динамические профили)")
    root.geometry("200x100")
    num_edges = config.SIM_GRID_SIZE; num_cells = num_edges - 1
    mock_x_edges = np.linspace(-100, 100, num_edges); mock_y_edges = np.linspace(-80, 80, num_edges)
    mock_coverage = np.zeros((num_cells, num_cells)) 
    mock_x_centers = (mock_x_edges[:-1] + mock_x_edges[1:]) / 2.0; mock_y_centers = (mock_y_edges[:-1] + mock_y_edges[1:]) / 2.0
    mock_YC, mock_XC = np.meshgrid(mock_y_centers, mock_x_centers, indexing='ij')
    mock_rr_centers = np.hypot(mock_XC, mock_YC)
    sigma_map = 50
    mock_coverage = 1000 * np.exp(-(mock_rr_centers**2 / (2 * sigma_map**2)))
    mock_coverage += np.random.rand(num_cells, num_cells) * 50
    mock_target_params_linear = {'target_type': config.TARGET_LINEAR, 'length': 200, 'width': 160, 'particles': 100000}
    mock_target_params_disk = {'target_type': config.TARGET_DISK, 'diameter': 150, 'particles': 100000}
    mock_vis_params = {'percent': True, 'logscale': False}
    def open_results_linear():
        print("Открытие окна результатов (Линейный)...")
        try: ResultsWindow(root, mock_coverage, mock_x_edges, mock_y_edges, mock_rr_centers, mock_target_params_linear, mock_vis_params)
        except Exception as e: print(f"Ошибка при открытии ResultsWindow: {e}"); traceback.print_exc()
    def open_results_disk():
        print("Открытие окна результатов (Диск)...")
        try: ResultsWindow(root, mock_coverage, mock_x_edges, mock_y_edges, mock_rr_centers, mock_target_params_disk, mock_vis_params)
        except Exception as e: print(f"Ошибка при открытии ResultsWindow: {e}"); traceback.print_exc()
    
    button_lin = ttk.Button(root, text="Показать (Линейный)", command=open_results_linear)
    button_lin.pack(pady=5)
    button_disk = ttk.Button(root, text="Показать (Диск)", command=open_results_disk)
    button_disk.pack(pady=5)
    root.mainloop()
