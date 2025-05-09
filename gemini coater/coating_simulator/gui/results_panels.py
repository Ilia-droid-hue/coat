# coding: utf-8
# Файл: coating_simulator/gui/results_panels.py
"""
Содержит классы для панелей, используемых в ResultsWindow.
Версия 12.12 - Скорректирован subplots_adjust для map_figure для лучшего отображения подписей.
"""
import tkinter as tk
from tkinter import ttk, messagebox

try:
    from .. import config 
except ImportError:
    class ConfigMock:
        TARGET_DISK = "диск"
        VIS_DEFAULT_PERCENT = True
        VIS_DEFAULT_LOGSCALE = False
    config = ConfigMock()
    print("ПРЕДУПРЕЖДЕНИЕ (results_panels.py): Используется ConfigMock, т.к. не удалось импортировать config.")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class SettingsPanel(ttk.Frame):
    """
    Левая панель настроек для окна результатов.
    """
    def __init__(self, master, recalculate_callback, export_excel_callback, 
                 calculate_mask_callback, load_profiles_callback, reconstruct_map_callback, 
                 *args, **kwargs):
        super().__init__(master, padding=(10, 10, 5, 5), *args, **kwargs)
        self.recalculate_callback = recalculate_callback
        self.export_excel_callback = export_excel_callback
        self.calculate_mask_callback = calculate_mask_callback
        self.load_profiles_callback = load_profiles_callback
        self.reconstruct_map_callback = reconstruct_map_callback

        for i in range(5): 
            self.rowconfigure(i, weight=0)

        self.uniformity_formulas = { 
            'U1': "U₁ = (Max-Min)/(Max+Min)", 'U2': "U₂ = (Max-Min)/Mean",
            'U3': "U₃ = StdDev/Mean (σ/t̄)", 'U4': "U₄ = Min/Max"
        }
        self.uniformity_method_var = tk.StringVar(value='U3')
        self.selected_formula_text_var = tk.StringVar(value=self.uniformity_formulas['U3'])
        self.profile_t_max_var = tk.StringVar(value="Max: -") 
        self.profile_t_min_var = tk.StringVar(value="Min: -")
        self.profile_t_mean_var = tk.StringVar(value="Mean: -")
        
        self.smoothing_method_var = tk.StringVar(value='Savitzky-Golay')
        self.savgol_window_var = tk.StringVar(value='11')
        self.savgol_polyorder_var = tk.StringVar(value='3')
        self.polyfit_degree_var = tk.StringVar(value='5')
        
        self.display_percent_var = tk.BooleanVar(value=config.VIS_DEFAULT_PERCENT)
        self.use_logscale_var = tk.BooleanVar(value=config.VIS_DEFAULT_LOGSCALE)
        self.show_raw_profile_var = tk.BooleanVar(value=False)
        
        self.auto_uniformity_mask_height_var = tk.StringVar(value='50.0')
        self.auto_uniformity_mode_var = tk.StringVar(value='Маска')
        
        self.loaded_files_text_var = tk.StringVar(value="Файлы не загружены")
        self.reconstruction_method_var = tk.StringVar(value="Линейная интерполяция")

        self.vcmd_int = (self.register(self._validate_positive_int), '%P')
        self.vcmd_odd_int = (self.register(self._validate_odd_positive_int), '%P')
        self.vcmd_float = (self.register(self._validate_positive_float), '%P')

        self._create_widgets()
        self._initial_ui_update()

    def _validate_positive_int(self, P):
        if P == "": return True
        try: return int(P) > 0
        except ValueError: return False

    def _validate_odd_positive_int(self, P):
        if P == "": return True
        try:
            val = int(P)
            return val > 0 and val % 2 != 0 and val >=3
        except ValueError: return False

    def _validate_positive_float(self, P):
        if P == "": return True
        if P == ".": return True
        if P.count('.') > 1: return False
        try: return float(P) >= 0
        except ValueError: return False

    def _is_scipy_available(self):
        try:
            from scipy.signal import savgol_filter # noqa F401
            return True
        except ImportError:
            return False

    def _create_widgets(self):
        pady_label_frame = (0, 5) 
        pady_widget_internal = 1 

        method_frame = ttk.LabelFrame(self, text="Расчет равномерности U (по профилю)", padding=5)
        method_frame.grid(row=0, column=0, sticky=tk.EW, pady=pady_label_frame)
        method_frame.columnconfigure(1, weight=1)
        ttk.Label(method_frame, text="Метод:").grid(row=0, column=0, padx=(0,5), pady=pady_widget_internal, sticky=tk.W)
        combo_method = ttk.Combobox(method_frame, textvariable=self.uniformity_method_var,
                                    values=list(self.uniformity_formulas.keys()), state='readonly', width=5)
        combo_method.grid(row=0, column=1, padx=5, pady=pady_widget_internal, sticky=tk.EW)
        combo_method.bind("<<ComboboxSelected>>", self._on_settings_change)
        self.label_formula_display = ttk.Label(method_frame, textvariable=self.selected_formula_text_var,
                                               font=('TkDefaultFont', 9), foreground="grey", wraplength=220)
        self.label_formula_display.grid(row=1, column=0, columnspan=2, padx=5, pady=(0,2), sticky=tk.W)
        stats_profile_frame = ttk.Frame(method_frame)
        stats_profile_frame.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=(2,0))
        stats_profile_frame.columnconfigure((0,1,2), weight=1)
        self.label_t_max = ttk.Label(stats_profile_frame, textvariable=self.profile_t_max_var)
        self.label_t_max.grid(row=0, column=0, sticky=tk.W)
        self.label_t_min = ttk.Label(stats_profile_frame, textvariable=self.profile_t_min_var)
        self.label_t_min.grid(row=0, column=1, sticky=tk.W)
        self.label_t_mean = ttk.Label(stats_profile_frame, textvariable=self.profile_t_mean_var)
        self.label_t_mean.grid(row=0, column=2, sticky=tk.W)

        profile_view_frame = ttk.LabelFrame(self, text="Настройки профиля и вида", padding=5)
        profile_view_frame.grid(row=1, column=0, sticky=tk.EW, pady=pady_label_frame)
        profile_view_frame.columnconfigure(1, weight=1)
        ttk.Label(profile_view_frame, text="Сглаживание:").grid(row=0, column=0, padx=(0,5), pady=pady_widget_internal, sticky=tk.W)
        smoothing_options = ["Без сглаживания", "Полином. аппрокс."]
        if self._is_scipy_available():
            smoothing_options.insert(0, "Savitzky-Golay")
            self.smoothing_method_var.set("Savitzky-Golay")
        else:
            self.smoothing_method_var.set("Полином. аппрокс.")
        combo_smoothing = ttk.Combobox(profile_view_frame, textvariable=self.smoothing_method_var,
                                       values=smoothing_options, state='readonly', width=18)
        combo_smoothing.grid(row=0, column=1, padx=5, pady=pady_widget_internal, sticky=tk.EW)
        combo_smoothing.bind("<<ComboboxSelected>>", self._on_settings_change)
        
        self.savgol_frame = ttk.Frame(profile_view_frame)
        ttk.Label(self.savgol_frame, text="Длина окна (нечет):").grid(row=0, column=0, padx=(0,2), pady=pady_widget_internal, sticky=tk.W)
        self.entry_savgol_window = ttk.Entry(self.savgol_frame, textvariable=self.savgol_window_var, width=5, validate='key', validatecommand=self.vcmd_odd_int)
        self.entry_savgol_window.grid(row=0, column=1, padx=2, pady=pady_widget_internal, sticky=tk.EW)
        ttk.Label(self.savgol_frame, text="Порядок полинома:").grid(row=1, column=0, padx=(0,2), pady=pady_widget_internal, sticky=tk.W)
        self.entry_savgol_polyorder = ttk.Entry(self.savgol_frame, textvariable=self.savgol_polyorder_var, width=5, validate='key', validatecommand=self.vcmd_int)
        self.entry_savgol_polyorder.grid(row=1, column=1, padx=2, pady=pady_widget_internal, sticky=tk.EW)
        
        self.polyfit_frame = ttk.Frame(profile_view_frame)
        ttk.Label(self.polyfit_frame, text="Степень полинома:").grid(row=0, column=0, padx=(0,2), pady=pady_widget_internal, sticky=tk.W)
        self.entry_polyfit_degree = ttk.Entry(self.polyfit_frame, textvariable=self.polyfit_degree_var, width=5, validate='key', validatecommand=self.vcmd_int)
        self.entry_polyfit_degree.grid(row=0, column=1, padx=2, pady=pady_widget_internal, sticky=tk.EW)
        
        self.entry_savgol_window.bind("<KeyRelease>", self._on_settings_change_entry)
        self.entry_savgol_polyorder.bind("<KeyRelease>", self._on_settings_change_entry)
        self.entry_polyfit_degree.bind("<KeyRelease>", self._on_settings_change_entry)

        chk_raw_profile = ttk.Checkbutton(profile_view_frame, text="Показать сырой профиль", variable=self.show_raw_profile_var, command=self.recalculate_callback)
        chk_raw_profile.grid(row=2, column=0, columnspan=2, pady=pady_widget_internal, sticky=tk.W)
        chk_percent = ttk.Checkbutton(profile_view_frame, text="Покрытие в %", variable=self.display_percent_var, command=self.recalculate_callback)
        chk_percent.grid(row=3, column=0, columnspan=2, pady=pady_widget_internal, sticky=tk.W)
        chk_logscale = ttk.Checkbutton(profile_view_frame, text="Лог. шкала (карта)", variable=self.use_logscale_var, command=self.recalculate_callback)
        chk_logscale.grid(row=4, column=0, columnspan=2, pady=pady_widget_internal, sticky=tk.W)

        auto_frame = ttk.LabelFrame(self, text="Авторавномерность", padding=5)
        auto_frame.grid(row=2, column=0, sticky=tk.EW, pady=pady_label_frame)
        auto_frame.columnconfigure(1, weight=1)
        ttk.Label(auto_frame, text="Метод:").grid(row=0, column=0, padx=(0,5), pady=pady_widget_internal, sticky=tk.W)
        self.combo_auto_mode = ttk.Combobox(auto_frame, textvariable=self.auto_uniformity_mode_var,
                                       values=['Маска', 'Источник'], state='readonly', width=10)
        self.combo_auto_mode.grid(row=0, column=1, padx=5, pady=pady_widget_internal, sticky=tk.EW)
        ttk.Label(auto_frame, text="Высота маски (мм):").grid(row=1, column=0, padx=(0,5), pady=pady_widget_internal, sticky=tk.W)
        self.entry_mask_height = ttk.Entry(auto_frame, textvariable=self.auto_uniformity_mask_height_var, width=7, validate='key', validatecommand=self.vcmd_float)
        self.entry_mask_height.grid(row=1, column=1, padx=5, pady=pady_widget_internal, sticky=tk.EW)
        self.btn_calc_mask = ttk.Button(auto_frame, text="Рассчитать", command=self.calculate_mask_callback)
        self.btn_calc_mask.grid(row=2, column=0, columnspan=2, pady=(3,0))

        inverse_problem_frame = ttk.LabelFrame(self, text="Обратная задача (по профилям)", padding=5)
        inverse_problem_frame.grid(row=3, column=0, sticky=tk.EW, pady=pady_label_frame)
        inverse_problem_frame.columnconfigure(0, weight=1) 
        inverse_problem_frame.columnconfigure(1, weight=1) 
        btn_load_profiles = ttk.Button(inverse_problem_frame, text="Загрузить профиль(и)", command=self.load_profiles_callback)
        btn_load_profiles.grid(row=0, column=0, columnspan=2, pady=pady_widget_internal, sticky=tk.EW)
        self.label_loaded_files = ttk.Label(inverse_problem_frame, textvariable=self.loaded_files_text_var, wraplength=220, justify=tk.LEFT)
        self.label_loaded_files.grid(row=1, column=0, columnspan=2, pady=pady_widget_internal, sticky=tk.W)
        ttk.Label(inverse_problem_frame, text="Метод реконстр.:").grid(row=2, column=0, pady=pady_widget_internal, sticky=tk.W)
        self.combo_reconstruction = ttk.Combobox(inverse_problem_frame, textvariable=self.reconstruction_method_var,
                                            values=["Линейная", "Кубический сплайн"], state='readonly', width=15)
        self.combo_reconstruction.grid(row=2, column=1, pady=pady_widget_internal, sticky=tk.EW)
        self.btn_reconstruct_map = ttk.Button(inverse_problem_frame, text="Построить карту по профилям", command=self._internal_reconstruct_map_callback)
        self.btn_reconstruct_map.grid(row=3, column=0, columnspan=2, pady=(3,0))

        control_panel = ttk.Frame(self, padding=(0, 2, 0, 0)) 
        control_panel.grid(row=4, column=0, sticky=tk.EW, pady=(5,0)) 
        control_panel.columnconfigure((0, 1), weight=1)
        btn_update = ttk.Button(control_panel, text="Обновить графики", command=self.recalculate_callback)
        btn_update.grid(row=0, column=0, padx=2, pady=(2,0), sticky=tk.EW) 
        self.btn_export_excel = ttk.Button(control_panel, text="Экспорт в Excel", command=self.export_excel_callback, state=tk.DISABLED)
        self.btn_export_excel.grid(row=0, column=1, padx=2, pady=(2,0), sticky=tk.EW)

    def _internal_reconstruct_map_callback(self):
        if self.reconstruct_map_callback:
            self.reconstruct_map_callback(self.reconstruction_method_var.get())

    def _initial_ui_update(self):
        self._update_formula_display_text_only_internal()
        self._update_smoothing_options_display_internal()

    def _on_settings_change(self, event=None):
        self._update_formula_display_text_only_internal() 
        self._update_smoothing_options_display_internal() 
        if self.recalculate_callback:
            self.recalculate_callback()

    def _on_settings_change_entry(self, event=None):
        if self.recalculate_callback:
            self.after_idle(self.recalculate_callback) 

    def _update_formula_display_text_only_internal(self): 
        selected_key = self.uniformity_method_var.get()
        formula_text = self.uniformity_formulas.get(selected_key, "Неизвестная формула")
        self.selected_formula_text_var.set(formula_text)

    def _update_smoothing_options_display_internal(self): 
        method = self.smoothing_method_var.get()
        options_start_row = 1 

        self.savgol_frame.grid_remove()
        self.polyfit_frame.grid_remove()

        if method == "Savitzky-Golay":
            if self._is_scipy_available():
                self.savgol_frame.grid(row=options_start_row, column=0, columnspan=2, sticky=tk.EW, padx=(10,0), pady=(3,0))
            else:
                print("Savitzky-Golay недоступен (SciPy)")
        elif method == "Полином. аппрокс.":
            self.polyfit_frame.grid(row=options_start_row, column=0, columnspan=2, sticky=tk.EW, padx=(10,0), pady=(3,0))
    
    def get_uniformity_method(self) -> str:
        return self.uniformity_method_var.get()

    def update_profile_stats(self, t_max: str, t_min: str, t_mean: str):
        self.profile_t_max_var.set(f"Max: {t_max}")
        self.profile_t_min_var.set(f"Min: {t_min}")
        self.profile_t_mean_var.set(f"Mean: {t_mean}")

    def get_view_settings(self) -> dict | None:
        smoothing_params_data = {'method': self.smoothing_method_var.get()}
        specific_params = {}
        try:
            if smoothing_params_data['method'] == "Savitzky-Golay":
                if not self._is_scipy_available():
                    smoothing_params_data['method'] = "Без сглаживания" 
                else:
                    win_str = self.savgol_window_var.get(); poly_str = self.savgol_polyorder_var.get()
                    if not win_str or not poly_str: raise ValueError("Параметры SavGol не могут быть пустыми")
                    specific_params['window_length'] = int(win_str)
                    specific_params['polyorder'] = int(poly_str)
                    if specific_params['window_length'] < 3 or specific_params['window_length'] % 2 == 0:
                        raise ValueError("Длина окна SavGol должна быть >=3 и нечетной")
                    if specific_params['polyorder'] >= specific_params['window_length']:
                        raise ValueError("Порядок полинома SavGol должен быть меньше длины окна")
            elif smoothing_params_data['method'] == "Полином. аппрокс.":
                deg_str = self.polyfit_degree_var.get()
                if not deg_str: raise ValueError("Степень полинома не может быть пустой")
                specific_params['degree'] = int(deg_str)
                if specific_params['degree'] < 1:
                    raise ValueError("Степень полинома должна быть >= 1")
            smoothing_params_data['params'] = specific_params
        except ValueError as e:
            messagebox.showerror("Ошибка ввода", f"Некорректное значение в настройках сглаживания: {e}", parent=self)
            return None 
        
        return {
            'smoothing': smoothing_params_data,
            'display_percent': self.display_percent_var.get(),
            'use_logscale': self.use_logscale_var.get(),
            'show_raw_profile': self.show_raw_profile_var.get()
        }

    def get_auto_uniformity_params(self) -> dict | None:
        try:
            return {
                'mode': self.auto_uniformity_mode_var.get(),
                'mask_height': float(self.auto_uniformity_mask_height_var.get())
            }
        except ValueError:
            messagebox.showerror("Ошибка ввода", "Некорректная высота маски для авторавномерности.", parent=self)
            return None
            
    def get_reconstruction_method(self) -> str: 
        return self.reconstruction_method_var.get()

    def update_loaded_files_text(self, text: str):
        self.loaded_files_text_var.set(text)

    def enable_export_button(self):
        self.btn_export_excel.config(state=tk.NORMAL)

    def disable_export_button(self):
        self.btn_export_excel.config(state=tk.DISABLED)


class PlotDisplayArea(ttk.Frame):
    """
    Панель для отображения графиков карты покрытия и профиля.
    """
    def __init__(self, master, plot_size_pixels=350, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.plot_size_pixels = plot_size_pixels
        
        self.columnconfigure(0, weight=1) 
        self.rowconfigure(0, weight=0)    # plots_container
        self.rowconfigure(1, weight=0)    # common_toolbar_frame

        self._create_widgets()

    def _create_widgets(self):
        gutter_size = 4 # Общий размер промежутка

        plots_container = ttk.Frame(self)
        plots_container.grid(row=0, column=0, sticky=tk.EW, pady=(0, gutter_size)) 
        plots_container.columnconfigure(0, weight=1) 
        plots_container.columnconfigure(1, weight=1) 

        plot_dpi = self.winfo_fpixels('1i') 
        if not isinstance(plot_dpi, (float, int)) or plot_dpi <=0: plot_dpi = 96 
        plot_size_inches = self.plot_size_pixels / plot_dpi

        map_plot_container = ttk.Frame(plots_container, width=self.plot_size_pixels, height=self.plot_size_pixels)
        map_plot_container.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, gutter_size // 2)) 
        map_plot_container.grid_propagate(False)
        self.map_figure = Figure(figsize=(plot_size_inches, plot_size_inches), dpi=plot_dpi)
        # Убираем subplots_adjust отсюда, он будет применяться в ResultsWindow
        # self.map_figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.95) 
        self.map_canvas = FigureCanvasTkAgg(self.map_figure, master=map_plot_container)
        self.map_canvas_widget = self.map_canvas.get_tk_widget()
        self.map_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        profile_plot_container = ttk.Frame(plots_container, width=self.plot_size_pixels, height=self.plot_size_pixels)
        profile_plot_container.grid(row=0, column=1, sticky=tk.NSEW, padx=(gutter_size // 2, 0)) 
        profile_plot_container.grid_propagate(False)
        self.profile_figure = Figure(figsize=(plot_size_inches, plot_size_inches), dpi=plot_dpi)
        self.profile_figure.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.92) 
        self.profile_canvas = FigureCanvasTkAgg(self.profile_figure, master=profile_plot_container)
        self.profile_canvas_widget = self.profile_canvas.get_tk_widget()
        self.profile_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        common_toolbar_frame = ttk.Frame(self)
        common_toolbar_frame.grid(row=1, column=0, sticky=tk.EW, pady=(0,0)) 
        self.toolbar = NavigationToolbar2Tk(self.map_canvas, common_toolbar_frame) 
        self.toolbar.update()

    def get_map_figure(self):
        return self.map_figure
    
    def get_profile_figure(self):
        return self.profile_figure

    def draw_canvases(self):
        self.map_canvas.draw_idle()
        self.profile_canvas.draw_idle()


class InfoDisplayArea(ttk.Frame):
    """
    Нижняя панель для отображения результатов U, информации по авторавномерности и обратной задаче.
    """
    def __init__(self, master, background_color, *args, **kwargs):
        super().__init__(master, padding=(0,2,0,0), style="InfoArea.TFrame", *args, **kwargs) 
        self.background_color = background_color

        self.columnconfigure(0, weight=1) 
        self.columnconfigure(1, weight=1) 
        self.columnconfigure(2, weight=1) 
        self.rowconfigure(0, weight=1)    

        self._create_widgets()

    def _create_widgets(self):
        lf_padding = (5,1) 
        label_pady = 1     
        label_font = ('TkDefaultFont', 9) 

        self.results_display_lf = ttk.LabelFrame(self, text="Результат U", padding=lf_padding)
        self.results_display_lf.grid(row=0, column=0, sticky=tk.NSEW, padx=(0,1)) 
        self.results_display_lf.rowconfigure(0, weight=1) 
        self.results_display_lf.columnconfigure(0, weight=1)
        self.label_uniformity_results = ttk.Label(self.results_display_lf, text="Нажмите 'Обновить графики'",
                                                  anchor=tk.NW, justify=tk.LEFT, wraplength=160, 
                                                  font=label_font, background=self.background_color)
        self.label_uniformity_results.grid(row=0, column=0, sticky=tk.NSEW, padx=3, pady=label_pady) 

        self.auto_uniformity_lf = ttk.LabelFrame(self, text="Авторавномерность", padding=lf_padding) 
        self.auto_uniformity_lf.grid(row=0, column=1, sticky=tk.NSEW, padx=1) 
        self.auto_uniformity_lf.rowconfigure(0, weight=1)
        self.auto_uniformity_lf.columnconfigure(0, weight=1)
        self.label_auto_uniformity_info = ttk.Label(self.auto_uniformity_lf, text="Результаты/статус...", 
                                                    anchor=tk.NW, justify=tk.LEFT, wraplength=160, font=label_font, background=self.background_color)
        self.label_auto_uniformity_info.grid(row=0, column=0, sticky=tk.NSEW, padx=3, pady=label_pady)

        self.inverse_problem_lf = ttk.LabelFrame(self, text="Обратная Задача", padding=lf_padding) 
        self.inverse_problem_lf.grid(row=0, column=2, sticky=tk.NSEW, padx=(1,0)) 
        self.inverse_problem_lf.rowconfigure(0, weight=1)
        self.inverse_problem_lf.columnconfigure(0, weight=1)
        self.label_inverse_problem_info = ttk.Label(self.inverse_problem_lf, text="Результаты/статус...", 
                                                    anchor=tk.NW, justify=tk.LEFT, wraplength=160, font=label_font, background=self.background_color)
        self.label_inverse_problem_info.grid(row=0, column=0, sticky=tk.NSEW, padx=3, pady=label_pady)

    def update_uniformity_results(self, text: str):
        self.label_uniformity_results.config(text=text)

    def update_auto_uniformity_info(self, text: str):
        self.label_auto_uniformity_info.config(text=text)

    def update_inverse_problem_info(self, text: str):
        self.label_inverse_problem_info.config(text=text)

