# coating_simulator_project/coating_simulator/gui/frames/emission.py
"""
Frame for configuring particle emission parameters.
Фрейм для настройки параметров эмиссии частиц.
Particle count input changed to base x 10^exponent format.
Particle input fields are now visible by default and base entry width is adjusted.
Specific distribution fields are now correctly placed on initialization.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import traceback
import math

# Используем относительные импорты
from ... import config # Импортируем константы

class EmissionFrame(ttk.LabelFrame):
    """
    LabelFrame containing widgets for emission configuration.
    LabelFrame, содержащий виджеты для конфигурации эмиссии.
    """
    def __init__(self, master, processing_frame_update_callback=None, **kwargs):
        """
        Initializes the EmissionFrame.
        Инициализирует EmissionFrame.
        """
        super().__init__(master, text="Параметры эмиссии", **kwargs)
        self.columnconfigure(1, weight=1) # Allow entry column to expand

        self._processing_update_callback = processing_frame_update_callback

        # Validation commands
        vcmd_float_base = (self.register(self._validate_float_positive), '%P') # Base must be > 0
        vcmd_int_exponent = (self.register(self._validate_int_non_negative), '%P') # Exponent >= 0
        vcmd_angle = (self.register(self._validate_full_angle), '%P')

        # --- Widgets ---
        # Row 0: Distribution Type
        ttk.Label(self, text="Тип распределения:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5), pady=2)
        self.combo_dist = ttk.Combobox(self, values=config.DISTRIBUTION_TYPES, state="readonly", width=20)
        self.combo_dist.set(config.DIST_GAUSSIAN)
        self.combo_dist.grid(row=0, column=1, sticky=tk.EW, pady=2)

        # Row 1: Full Emission Angle (Theta_full)
        ttk.Label(self, text="Полный угол эмиссии (°):").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=2)
        self.entry_full_angle = ttk.Entry(self, validate='key', validatecommand=vcmd_angle)
        self.entry_full_angle.insert(0, str(config.DEFAULT_EMISSION_PARAMS["max_theta"] * 2.0))
        self.entry_full_angle.grid(row=1, column=1, sticky=tk.EW, pady=2)

        # Row 2: Particle Count Input (Base x 10^Exponent) - ALWAYS AT ROW 2
        self.label_parts = ttk.Label(self, text="Частиц:")
        self.particles_input_frame = ttk.Frame(self) # Frame to hold base, label, and exponent entries

        self.entry_particles_base = ttk.Entry(self.particles_input_frame, width=4, validate='key', validatecommand=vcmd_float_base)
        self.label_times_ten = ttk.Label(self.particles_input_frame, text="x 10^")
        self.entry_particles_exponent = ttk.Entry(self.particles_input_frame, width=3, validate='key', validatecommand=vcmd_int_exponent)

        # Set default values for particle count
        default_total_particles = int(config.DEFAULT_EMISSION_PARAMS["particles"])
        if default_total_particles > 0:
            default_exponent = 0
            default_base = float(default_total_particles)
            if default_total_particles > 0: # Ensure base calculation only if positive
                # Приводим базу к виду X.Y... или X, где X одна цифра, если возможно, или X.Y
                # Например, 10000 -> 1e4, 25000 -> 2.5e4, 500 -> 5e2, 1 -> 1e0
                if default_base >= 1.0: # Только если >= 1, иначе оставляем как есть (e.g. 0.5)
                    default_exponent = math.floor(math.log10(default_base))
                    default_base /= (10**default_exponent)

            # Форматируем для удаления ".0" если число целое, и ограничиваем точность
            if default_base == int(default_base):
                base_str = str(int(default_base))
            else:
                base_str = f"{default_base:.2f}".rstrip('0').rstrip('.') # Оставляем до 2 знаков после запятой

            self.entry_particles_base.insert(0, base_str)
            self.entry_particles_exponent.insert(0, str(default_exponent))
        else: 
            self.entry_particles_base.insert(0, "1") # Default to 1 x 10^4 if config is invalid
            self.entry_particles_exponent.insert(0, "4")

        self.entry_particles_base.pack(side=tk.LEFT, padx=(0,1))
        self.label_times_ten.pack(side=tk.LEFT, padx=1)
        self.entry_particles_exponent.pack(side=tk.LEFT, padx=(1,0))
        
        # Place particle widgets (label and frame) at fixed row 2
        self.label_parts.grid(row=2, column=0, sticky=tk.W, padx=(0, 5), pady=2)
        self.particles_input_frame.grid(row=2, column=1, sticky=tk.EW, pady=2)

        # --- Distribution Specific Widgets (created but not gridded initially) ---
        # Specific widgets will start from row 3
        self.specific_widgets = {}

        self.label_sigma = ttk.Label(self, text="Sigma (σ), °:")
        self.entry_sigma = ttk.Entry(self, validate='key', validatecommand=vcmd_float_base)
        self.entry_sigma.insert(0, str(config.DEFAULT_EMISSION_PARAMS["sigma"]))
        self.specific_widgets[config.DIST_GAUSSIAN] = [
            (self.label_sigma, self.entry_sigma)
        ]

        self.label_m = ttk.Label(self, text="Экспонента m:")
        self.entry_m = ttk.Entry(self, validate='key', validatecommand=vcmd_int_exponent)
        self.entry_m.insert(0, str(config.DEFAULT_EMISSION_PARAMS["m_exp"]))
        self.specific_widgets[config.DIST_COSINE_POWER] = [
            (self.label_m, self.entry_m)
        ]
        self.specific_widgets[config.DIST_UNIFORM_SOLID] = []

        # --- Bindings and Initial State ---
        self.combo_dist.bind("<<ComboboxSelected>>", self._update_fields)
        
        # Initial call to place specific widgets correctly based on default distribution
        self._update_fields()
        # The ProcessingFrame will call show/hide_particles_entry as needed,
        # which in turn calls _update_fields again if visibility changes.

    def _validate_float_positive(self, P):
        if P == "": return True
        if P == ".": return True
        if P.count('.') > 1: return False
        try:
            val = float(P)
            return val > 0 
        except ValueError: return False

    def _validate_int_non_negative(self, P):
        if P == "": return True
        try:
            val = int(P)
            return val >= 0
        except ValueError: return False

    def _validate_full_angle(self, P):
        if P == "": return True
        if P == ".": return True
        if P.count('.') > 1: return False
        try:
            val = float(P)
            return 0 <= val <= 180
        except ValueError: return False

    def _update_fields(self, event=None):
        selected_dist = self.combo_dist.get()
        
        for dist_type_key in self.specific_widgets:
            for label, entry in self.specific_widgets[dist_type_key]:
                label.grid_remove()
                entry.grid_remove()

        # Specific widgets always start at row 3, below particle input
        current_row_for_specifics = 3 

        if selected_dist in self.specific_widgets:
            for label, entry in self.specific_widgets[selected_dist]:
                label.grid(row=current_row_for_specifics, column=0, sticky=tk.W, padx=(0, 5), pady=2)
                entry.grid(row=current_row_for_specifics, column=1, sticky=tk.EW, pady=2)
                current_row_for_specifics += 1
        
        # This callback is for ProcessingFrame to know if it needs to adjust its layout
        # (e.g. if EmissionFrame suddenly takes more or less vertical space due to its own widgets)
        # However, since particle input is now always "visible" from EmissionFrame's perspective
        # (its visibility is controlled by ProcessingFrame), and specific widgets are below it,
        # the height of EmissionFrame changes predictably.
        if self._processing_update_callback:
             try: self._processing_update_callback()
             except Exception: traceback.print_exc()

    def get_params(self) -> dict:
        params = {}
        dist_type = self.combo_dist.get()
        params['dist_type'] = dist_type

        try:
            full_angle_deg_str = self.entry_full_angle.get()
            if not full_angle_deg_str: raise ValueError("Полный угол эмиссии не может быть пустым.")
            full_angle_deg = float(full_angle_deg_str)
            if not (0 <= full_angle_deg <= 180):
                 raise ValueError("Полный угол эмиссии должен быть между 0 и 180 градусами.")
            params['max_theta'] = full_angle_deg / 2.0

            base_str = self.entry_particles_base.get()
            exp_str = self.entry_particles_exponent.get()
            if not base_str or not exp_str:
                raise ValueError("Значение и степень для числа частиц не могут быть пустыми.")

            base_val = float(base_str)
            exp_val = int(exp_str)

            if base_val <= 0:
                raise ValueError("Базовое значение для числа частиц должно быть положительным.")
            if exp_val < 0:
                raise ValueError("Степень для числа частиц не может быть отрицательной.")

            total_particles = int(base_val * (10**exp_val))
            if total_particles <= 0:
                 raise ValueError("Итоговое число частиц должно быть положительным целым числом.")
            params['particles'] = total_particles

            if dist_type == config.DIST_GAUSSIAN:
                sigma_str = self.entry_sigma.get()
                if not sigma_str: raise ValueError("Sigma не может быть пустой.")
                params['sigma'] = float(sigma_str)
                if params['sigma'] <= 0:
                     raise ValueError("Sigma (σ) для Gaussian beam должна быть положительной.")
                if params['max_theta'] > 1e-6 and params['sigma'] > params['max_theta']:
                     print(f"Предупреждение: Sigma ({params['sigma']}°) больше половины полного угла ({params['max_theta']}°).")
            elif dist_type == config.DIST_COSINE_POWER:
                m_str = self.entry_m.get()
                if not m_str: raise ValueError("Экспонента m не может быть пустой.")
                params['m_exp'] = float(m_str)
                if params['m_exp'] < 0:
                     raise ValueError("Экспонента m для Cosine-power не может быть отрицательной.")

        except ValueError as e:
            messagebox.showerror("Ошибка ввода (Эмиссия)", f"Некорректное значение в параметрах эмиссии: {e}", parent=self)
            raise ValueError(f"Invalid emission parameter: {e}") from e
        except Exception as e:
             messagebox.showerror("Ошибка (Эмиссия)", f"Неожиданная ошибка при чтении параметров эмиссии: {e}", parent=self)
             raise RuntimeError(f"Unexpected error getting emission params: {e}") from e
        return params

    def show_particles_entry(self): # Убран параметр 'row'
        """Grids the particle count widgets if they are not already visible."""
        if not self.label_parts.winfo_ismapped():
            self.label_parts.grid(row=2, column=0, sticky=tk.W, padx=(0, 5), pady=2)
        if not self.particles_input_frame.winfo_ismapped():
            self.particles_input_frame.grid(row=2, column=1, sticky=tk.EW, pady=2)
        self._update_fields() 

    def hide_particles_entry(self):
        """Removes the particle count widgets from the grid."""
        self.label_parts.grid_remove()
        self.particles_input_frame.grid_remove()
        self._update_fields()
