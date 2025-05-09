# coding: utf-8
# coating_simulator_project/coating_simulator/gui/results.py
"""
Toplevel window for displaying simulation results with interactive uniformity controls,
and a section for inverse problem solving (reconstructing map from profiles).
(GUI Skeleton with Inverse Problem Section)
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import traceback
import math
import os # Для работы с именами файлов

# Используем относительные импорты
try:
    from ..visualization.plot import (plot_simulation_results as plot_on_figure,
                                      calculate_uniformity_stats,
                                      format_uniformity_stats,
                                      _smooth_profile as smooth_profile_data)
    PLOT_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"ОШИБКА ИМПОРТА из plot.py: {e}")
    PLOT_MODULE_AVAILABLE = False
    # Заглушки
    def plot_on_figure(*args, **kwargs): print("plot_on_figure (заглушка)")
    def calculate_uniformity_stats(*args, **kwargs): print("calculate_uniformity_stats (заглушка)"); return {}
    def format_uniformity_stats(*args, **kwargs): print("format_uniformity_stats (заглушка)"); return "Ошибка импорта plot.py"
    def smooth_profile_data(coords, data, method, params): print("smooth_profile_data (заглушка)"); return data

from .. import config

# Встраивание Matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.patches as patches

class ResultsWindow(tk.Toplevel):
    """
    Окно для отображения результатов симуляции и анализа равномерности,
    включая секцию для решения обратной задачи.
    (Каркас GUI v10 - Секция Обратной Задачи)
    """
    def __init__(self, parent, coverage_map: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, radius_grid: np.ndarray,
                 target_params: dict, vis_params: dict):
        """ Инициализирует ResultsWindow (Каркас GUI). """
        super().__init__(parent)
        self.title("Результаты: Анализ, Сглаживание и Обратная Задача")
        self.geometry("1200x720") # Немного увеличим высоту для новой секции
        self.minsize(950, 650)

        # Сохраняем данные симуляции
        self.simulation_coverage_map_raw = coverage_map
        self.simulation_x_coords = x_coords
        self.simulation_y_coords = y_coords
        self.simulation_radius_grid = radius_grid
        self.simulation_target_params = target_params # Параметры цели из симуляции
        self.current_vis_params = vis_params.copy()

        # Данные для обратной задачи
        self.loaded_profiles_data = [] # Список для хранения данных загруженных профилей
        self.reconstructed_coverage_map = None
        self.reconstructed_x_coords = None
        self.reconstructed_y_coords = None
        self.reconstructed_radius_grid = None


        self.current_target_type = self.simulation_target_params.get('target_type', config.TARGET_DISK)

        self.uniformity_formulas = {
            'U1': "U₁ = (Max-Min)/(Max+Min)", 'U2': "U₂ = (Max-Min)/Mean",
            'U3': "U₃ = StdDev/Mean (σ/t̄)", 'U4': "U₄ = Min/Max"
        }

        # --- Переменные Tkinter ---
        self.uniformity_method_var = tk.StringVar(value='U3')
        self.selected_formula_text_var = tk.StringVar(value=self.uniformity_formulas['U3'])
        self.profile_t_max_var = tk.StringVar(value="Max: -")
        self.profile_t_min_var = tk.StringVar(value="Min: -")
        self.profile_t_mean_var = tk.StringVar(value="Mean: -")
        self.smoothing_method_var = tk.StringVar(value='Savitzky-Golay')
        self.savgol_window_var = tk.StringVar(value='11')
        self.savgol_polyorder_var = tk.StringVar(value='3')
        self.polyfit_degree_var = tk.StringVar(value='5')
        self.display_percent_var = tk.BooleanVar(value=self.current_vis_params.get('percent', config.VIS_DEFAULT_PERCENT))
        self.use_logscale_var = tk.BooleanVar(value=self.current_vis_params.get('logscale', config.VIS_DEFAULT_LOGSCALE))
        self.show_raw_profile_var = tk.BooleanVar(value=False)
        # Авторавномерность
        self.auto_uniformity_mask_height_var = tk.StringVar(value='50.0')
        self.auto_uniformity_mode_var = tk.StringVar(value='Маска')
        # Обратная задача
        self.loaded_files_text_var = tk.StringVar(value="Файлы не загружены")
        self.reconstruction_method_var = tk.StringVar(value="Линейная интерполяция") # Пример

        # Валидация
        vcmd_int = (self.register(self._validate_positive_int), '%P')
        vcmd_odd_int = (self.register(self._validate_odd_positive_int), '%P')
        vcmd_float = (self.register(self._validate_positive_float), '%P')

        # --- Структура окна ---
        self.columnconfigure(0, weight=0, minsize=280)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # --- Левая панель (Настройки) ---
        settings_panel = ttk.Frame(self, padding=(10, 10, 5, 10))
        settings_panel.grid(row=0, column=0, sticky=tk.NSEW)
        # Обновляем rowconfigure для новой секции
        settings_panel.rowconfigure(4, weight=1) # Результат U
        settings_panel.rowconfigure(5, weight=0) # Панель управления

        # --- 1. Метод расчета U ---
        method_frame = ttk.LabelFrame(settings_panel, text="Расчет равномерности U (по профилю)", padding=5)
        method_frame.grid(row=0, column=0, sticky=tk.EW, pady=(0, 7))
        # ... (код для method_frame без изменений) ...
        method_frame.columnconfigure(1, weight=1)
        ttk.Label(method_frame, text="Метод:").grid(row=0, column=0, padx=(0,5), pady=2, sticky=tk.W)
        combo_method = ttk.Combobox(method_frame, textvariable=self.uniformity_method_var,
                                    values=list(self.uniformity_formulas.keys()), state='readonly', width=5)
        combo_method.grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)
        combo_method.bind("<<ComboboxSelected>>", self._update_formula_display_text_only)
        self.label_formula_display = ttk.Label(method_frame, textvariable=self.selected_formula_text_var,
                                               font=('TkDefaultFont', 9), foreground="grey", wraplength=220)
        self.label_formula_display.grid(row=1, column=0, columnspan=2, padx=5, pady=(0,3), sticky=tk.W)
        stats_profile_frame = ttk.Frame(method_frame)
        stats_profile_frame.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=(3,0))
        stats_profile_frame.columnconfigure((0,1,2), weight=1)
        ttk.Label(stats_profile_frame, textvariable=self.profile_t_max_var).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(stats_profile_frame, textvariable=self.profile_t_min_var).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(stats_profile_frame, textvariable=self.profile_t_mean_var).grid(row=0, column=2, sticky=tk.W)


        # --- 2. Настройки профиля и вида ---
        profile_view_frame = ttk.LabelFrame(settings_panel, text="Настройки профиля и вида", padding=5)
        profile_view_frame.grid(row=1, column=0, sticky=tk.EW, pady=(0,7))
        # ... (код для profile_view_frame без изменений) ...
        profile_view_frame.columnconfigure(1, weight=1)
        ttk.Label(profile_view_frame, text="Сглаживание:").grid(row=0, column=0, padx=(0,5), pady=2, sticky=tk.W)
        smoothing_options = ["Без сглаживания", "Полином. аппрокс."]
        if self._is_scipy_available(): smoothing_options.insert(0, "Savitzky-Golay"); self.smoothing_method_var.set("Savitzky-Golay")
        else: self.smoothing_method_var.set("Полином. аппрокс.")
        combo_smoothing = ttk.Combobox(profile_view_frame, textvariable=self.smoothing_method_var,
                                       values=smoothing_options, state='readonly', width=18)
        combo_smoothing.grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)
        combo_smoothing.bind("<<ComboboxSelected>>", self._update_smoothing_options_display)
        self.savgol_frame = ttk.Frame(profile_view_frame)
        ttk.Label(self.savgol_frame, text="Длина окна (нечет):").grid(row=0, column=0, padx=(0,2), pady=1, sticky=tk.W)
        self.entry_savgol_window = ttk.Entry(self.savgol_frame, textvariable=self.savgol_window_var, width=5, validate='key', validatecommand=vcmd_odd_int)
        self.entry_savgol_window.grid(row=0, column=1, padx=2, pady=1, sticky=tk.EW)
        ttk.Label(self.savgol_frame, text="Порядок полинома:").grid(row=1, column=0, padx=(0,2), pady=1, sticky=tk.W)
        self.entry_savgol_polyorder = ttk.Entry(self.savgol_frame, textvariable=self.savgol_polyorder_var, width=5, validate='key', validatecommand=vcmd_int)
        self.entry_savgol_polyorder.grid(row=1, column=1, padx=2, pady=1, sticky=tk.EW)
        self.polyfit_frame = ttk.Frame(profile_view_frame)
        ttk.Label(self.polyfit_frame, text="Степень полинома:").grid(row=0, column=0, padx=(0,2), pady=1, sticky=tk.W)
        self.entry_polyfit_degree = ttk.Entry(self.polyfit_frame, textvariable=self.polyfit_degree_var, width=5, validate='key', validatecommand=vcmd_int)
        self.entry_polyfit_degree.grid(row=0, column=1, padx=2, pady=1, sticky=tk.EW)
        chk_raw_profile = ttk.Checkbutton(profile_view_frame, text="Показать сырой профиль", variable=self.show_raw_profile_var)
        chk_raw_profile.grid(row=2, column=0, columnspan=2, pady=2, sticky=tk.W)
        chk_percent = ttk.Checkbutton(profile_view_frame, text="Покрытие в %", variable=self.display_percent_var)
        chk_percent.grid(row=3, column=0, columnspan=2, pady=2, sticky=tk.W)
        chk_logscale = ttk.Checkbutton(profile_view_frame, text="Лог. шкала (карта)", variable=self.use_logscale_var)
        chk_logscale.grid(row=4, column=0, columnspan=2, pady=2, sticky=tk.W)


        # --- 3. Авторавномерность (Заглушка) ---
        auto_frame = ttk.LabelFrame(settings_panel, text="Авторавномерность", padding=5)
        auto_frame.grid(row=2, column=0, sticky=tk.EW, pady=(0, 10))
        # ... (код для auto_frame без изменений) ...
        auto_frame.columnconfigure(1, weight=1)
        ttk.Label(auto_frame, text="Метод:").grid(row=0, column=0, padx=(0,5), pady=2, sticky=tk.W)
        combo_auto_mode = ttk.Combobox(auto_frame, textvariable=self.auto_uniformity_mode_var,
                                       values=['Маска', 'Источник'], state='readonly', width=10)
        combo_auto_mode.grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)
        ttk.Label(auto_frame, text="Высота маски (мм):").grid(row=1, column=0, padx=(0,5), pady=2, sticky=tk.W)
        entry_mask_height = ttk.Entry(auto_frame, textvariable=self.auto_uniformity_mask_height_var, width=7, validate='key', validatecommand=vcmd_float)
        entry_mask_height.grid(row=1, column=1, padx=5, pady=2, sticky=tk.EW)
        btn_calc_mask = ttk.Button(auto_frame, text="Рассчитать", command=self._placeholder_calculate_mask)
        btn_calc_mask.grid(row=2, column=0, columnspan=2, pady=5)


        # --- НОВАЯ СЕКЦИЯ: Решение обратной задачи ---
        inverse_problem_frame = ttk.LabelFrame(settings_panel, text="Обратная задача (по профилям)", padding=5)
        inverse_problem_frame.grid(row=3, column=0, sticky=tk.EW, pady=(0,10)) # Размещаем перед Результатом U
        inverse_problem_frame.columnconfigure(0, weight=1) # Кнопки будут растягиваться

        btn_load_profiles = ttk.Button(inverse_problem_frame, text="Загрузить профиль(и)", command=self._placeholder_load_profiles)
        btn_load_profiles.grid(row=0, column=0, columnspan=2, pady=3, sticky=tk.EW)

        self.label_loaded_files = ttk.Label(inverse_problem_frame, textvariable=self.loaded_files_text_var, wraplength=220, justify=tk.LEFT)
        self.label_loaded_files.grid(row=1, column=0, columnspan=2, pady=3, sticky=tk.W)

        # (Опционально) Параметры реконструкции
        ttk.Label(inverse_problem_frame, text="Метод реконстр.:").grid(row=2, column=0, pady=3, sticky=tk.W)
        combo_reconstruction = ttk.Combobox(inverse_problem_frame, textvariable=self.reconstruction_method_var,
                                            values=["Линейная", "Кубический сплайн"], state='readonly', width=15) # Пример
        combo_reconstruction.grid(row=2, column=1, pady=3, sticky=tk.EW)


        btn_reconstruct_map = ttk.Button(inverse_problem_frame, text="Построить карту по профилям", command=self._placeholder_reconstruct_map)
        btn_reconstruct_map.grid(row=3, column=0, columnspan=2, pady=(5,2), sticky=tk.EW)
        # --------------------------------------------

        # --- 4. Результат равномерности --- (Теперь row=4)
        results_display_frame = ttk.LabelFrame(settings_panel, text="Результат U", padding=10)
        results_display_frame.grid(row=4, column=0, sticky=tk.NSEW, pady=(0, 10))
        self.label_uniformity_results = ttk.Label(results_display_frame, text="Нажмите 'Обновить'",
                                                  anchor=tk.NW, justify=tk.LEFT, wraplength=200,
                                                  font=('TkDefaultFont', 11))
        self.label_uniformity_results.pack(fill=tk.BOTH, expand=True)

        # --- 5. Панель управления --- (Теперь row=5)
        control_panel = ttk.Frame(settings_panel, padding=(0, 5, 0, 0))
        control_panel.grid(row=5, column=0, sticky=tk.EW)
        control_panel.columnconfigure((0, 1), weight=1)
        btn_update = ttk.Button(control_panel, text="Обновить", command=self._placeholder_recalculate_and_redraw)
        btn_update.grid(row=0, column=0, padx=2, pady=(5,0), sticky=tk.EW)
        self.btn_export_excel = ttk.Button(control_panel, text="Экспорт в Excel", command=self._placeholder_export_excel, state=tk.DISABLED)
        self.btn_export_excel.grid(row=0, column=1, padx=2, pady=(5,0), sticky=tk.EW)

        # --- Правая панель (Графики) ---
        plots_area_frame = ttk.Frame(self)
        plots_area_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=5, pady=5)
        plots_area_frame.columnconfigure(0, weight=1)
        plots_area_frame.columnconfigure(1, weight=1)
        plots_area_frame.rowconfigure(0, weight=0)
        plots_area_frame.rowconfigure(1, weight=0)
        plots_area_frame.rowconfigure(2, weight=1)

        plot_size_pixels = 380
        plot_dpi = 90
        plot_size_inches = plot_size_pixels / plot_dpi

        map_plot_container = ttk.Frame(plots_area_frame, width=plot_size_pixels, height=plot_size_pixels)
        map_plot_container.grid(row=0, column=0, sticky=tk.NSEW, padx=(0,5))
        map_plot_container.grid_propagate(False)
        self.map_figure = Figure(figsize=(plot_size_inches, plot_size_inches), dpi=plot_dpi)
        self.map_canvas = FigureCanvasTkAgg(self.map_figure, master=map_plot_container)
        self.map_canvas_widget = self.map_canvas.get_tk_widget()
        self.map_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        profile_plot_container = ttk.Frame(plots_area_frame, width=plot_size_pixels, height=plot_size_pixels)
        profile_plot_container.grid(row=0, column=1, sticky=tk.NSEW, padx=(5,0))
        profile_plot_container.grid_propagate(False)
        self.profile_figure = Figure(figsize=(plot_size_inches, plot_size_inches), dpi=plot_dpi)
        self.profile_canvas = FigureCanvasTkAgg(self.profile_figure, master=profile_plot_container)
        self.profile_canvas_widget = self.profile_canvas.get_tk_widget()
        self.profile_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        common_toolbar_frame = ttk.Frame(plots_area_frame)
        common_toolbar_frame.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(5,0))
        self.map_toolbar = NavigationToolbar2Tk(self.map_canvas, common_toolbar_frame) # Тулбар для карты
        self.map_toolbar.update()
        # Можно добавить второй тулбар для профиля, если нужно
        # self.profile_toolbar = NavigationToolbar2Tk(self.profile_canvas, common_toolbar_frame)
        # self.profile_toolbar.pack(side=tk.LEFT, expand=True, fill=tk.X) # Пример

        ttk.Frame(plots_area_frame).grid(row=2, column=0, columnspan=2, sticky=tk.NSEW) # Свободное место

        # --- Инициализация ---
        self._update_formula_display_text_only()
        self._update_smoothing_options_display()
        if PLOT_MODULE_AVAILABLE:
            self.after(50, self._placeholder_recalculate_and_redraw)
        else:
            messagebox.showerror("Ошибка импорта", "Не удалось загрузить модуль визуализации (plot.py).\nГрафики не будут отображены.", parent=self)
            self.label_uniformity_results.config(text="Ошибка импорта plot.py")

    # --- Функции валидации ---
    def _validate_positive_int(self, P):
        if P == "": return True
        try: return int(P) > 0
        except ValueError: return False

    def _validate_odd_positive_int(self, P):
        if P == "": return True
        try: val = int(P); return val > 0 and val % 2 != 0 and val >=3
        except ValueError: return False

    def _validate_positive_float(self, P):
        if P == "": return True;
        if P == ".": return True;
        if P.count('.') > 1: return False
        try: return float(P) >= 0
        except ValueError: return False

    def _validate_strictly_positive_float(self, P):
        if P == "": return True;
        if P == ".": return True;
        if P.count('.') > 1: return False
        try: return float(P) > 0
        except ValueError: return False

    def _is_scipy_available(self):
        try: from scipy.signal import savgol_filter; return True
        except ImportError: return False

    # --- Обновление GUI ---
    def _update_formula_display_text_only(self, event=None):
        selected_key = self.uniformity_method_var.get()
        formula_text = self.uniformity_formulas.get(selected_key, "Неизвестная формула")
        self.selected_formula_text_var.set(formula_text)

    def _update_smoothing_options_display(self, event=None):
        method = self.smoothing_method_var.get()
        parent_frame_for_smoothing_options = self.savgol_frame.master
        current_row_in_parent = 1

        if method == "Savitzky-Golay":
            if self._is_scipy_available():
                self.savgol_frame.grid(row=current_row_in_parent, column=0, columnspan=2, sticky=tk.EW, padx=(10,0))
            else:
                self.savgol_frame.grid_remove(); print("Savitzky-Golay недоступен (SciPy)")
            self.polyfit_frame.grid_remove()
        elif method == "Полином. аппрокс.":
            self.polyfit_frame.grid(row=current_row_in_parent, column=0, columnspan=2, sticky=tk.EW, padx=(10,0))
            self.savgol_frame.grid_remove()
        else: # "Без сглаживания"
            self.savgol_frame.grid_remove()
            self.polyfit_frame.grid_remove()

    # --- Функции-заглушки ---
    def _placeholder_recalculate_and_redraw(self):
        print("--- Обновление статистики и вида (Заглушка) ---")
        smoothing_params = self._get_current_smoothing_params()
        if smoothing_params is None:
            self.label_uniformity_results.config(text="Ошибка в параметрах сглаживания")
            return
        selected_method_U = self.uniformity_method_var.get()
        print(f"Метод U: {selected_method_U}, Сглаживание: {smoothing_params}")
        self.label_uniformity_results.config(text=f"Расчет для {selected_method_U}...")

        if hasattr(self, 'map_figure') and hasattr(self, 'profile_figure') and PLOT_MODULE_AVAILABLE:
            self.map_figure.clear(); self.profile_figure.clear()
            ax_map = self.map_figure.add_subplot(111)
            ax_profile = self.profile_figure.add_subplot(111)
            ax_map.set_title("Карта (заглушка)"); ax_map.grid(True)
            ax_profile.set_title("Профиль (заглушка)"); ax_profile.grid(True)
            self.map_figure.tight_layout(); self.profile_figure.tight_layout()
            if hasattr(self, 'map_canvas'): self.map_canvas.draw()
            if hasattr(self, 'profile_canvas'): self.profile_canvas.draw()
        messagebox.showinfo("Заглушка", "Функция обновления еще не реализована.", parent=self)

    def _placeholder_calculate_mask(self):
         messagebox.showinfo("Заглушка", "Функция расчета маски авторавномерности еще не реализована.", parent=self)

    def _placeholder_export_excel(self):
         messagebox.showinfo("Заглушка", "Функция экспорта в Excel еще не реализована.", parent=self)

    def _placeholder_load_profiles(self):
        filepaths = filedialog.askopenfilenames(
            title="Выберите файлы профилей (.csv, .txt)",
            filetypes=[("CSV файлы", "*.csv"), ("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")]
        )
        if filepaths:
            self.loaded_profiles_data = [] # Очищаем старые
            filenames = [os.path.basename(fp) for fp in filepaths]
            self.loaded_files_text_var.set("Загружено: " + ", ".join(filenames) if filenames else "Файлы не загружены")
            # Здесь будет логика чтения файлов
            print(f"Загружены файлы: {filepaths}")
            messagebox.showinfo("Загрузка", f"Выбрано файлов: {len(filepaths)}.\nЛогика обработки еще не реализована.", parent=self)
        else:
            self.loaded_files_text_var.set("Загрузка отменена")

    def _placeholder_reconstruct_map(self):
        if not self.loaded_profiles_data: # Проверка, что есть данные
            messagebox.showwarning("Обратная задача", "Сначала загрузите файлы профилей.", parent=self)
            return
        reconstr_method = self.reconstruction_method_var.get()
        print(f"--- Построение карты по профилям (Заглушка) ---")
        print(f"Используется метод: {reconstr_method}")
        print(f"Данные: {self.loaded_profiles_data}") # Позже здесь будут реальные данные
        # Здесь будет вызов реальной логики реконструкции и затем обновление графиков
        # self._placeholder_recalculate_and_redraw() # Для обновления с новыми данными
        messagebox.showinfo("Заглушка", f"Функция построения карты по профилям ({reconstr_method}) еще не реализована.", parent=self)


    # --- Вспомогательные методы ---
    def _get_current_smoothing_params(self) -> dict:
        # (Код без изменений)
        params = {'method': self.smoothing_method_var.get()}
        specific_params = {}
        try:
            if params['method'] == "Savitzky-Golay":
                if not self._is_scipy_available(): return {'method': "Без сглаживания", 'params': {}}
                win_str = self.savgol_window_var.get(); poly_str = self.savgol_polyorder_var.get()
                specific_params['window_length'] = int(win_str) if win_str else 11
                specific_params['polyorder'] = int(poly_str) if poly_str else 3
                if specific_params['window_length'] < 3 or specific_params['window_length'] % 2 == 0: raise ValueError("Длина окна SavGol >=3 и нечетное")
                if specific_params['polyorder'] >= specific_params['window_length']: raise ValueError("Порядок полинома SavGol < длины окна")
            elif params['method'] == "Полином. аппрокс.":
                deg_str = self.polyfit_degree_var.get()
                specific_params['degree'] = int(deg_str) if deg_str else 5
                if specific_params['degree'] < 1: raise ValueError("Степень полинома >= 1")
            params['params'] = specific_params
        except ValueError as e:
            messagebox.showerror("Ошибка ввода", f"Некорректное значение в настройках сглаживания: {e}", parent=self)
            return None
        return params

# Example usage
if __name__ == '__main__':
    root = tk.Tk()
    root.title("Тест ResultsWindow GUI Inverse Problem v1")
    root.geometry("200x100")
    mock_coverage = np.zeros((10,10)); mock_x = np.linspace(-10, 10, 10)
    mock_y = np.linspace(-10, 10, 10); mock_rr = np.zeros((10,10))
    mock_target_params = {'target_type': config.TARGET_DISK, 'diameter': 20}
    mock_vis_params = {'percent': True, 'logscale': False}
    def open_results():
        print("Открытие окна результатов (GUI Inverse Problem v1)...")
        try: ResultsWindow(root, mock_coverage, mock_x, mock_y, mock_rr, mock_target_params, mock_vis_params)
        except Exception as e: print(f"Ошибка: {e}"); traceback.print_exc()
    button = ttk.Button(root, text="Показать окно (GUI Inverse Problem v1)", command=open_results)
    button.pack(pady=20)
    root.mainloop()
