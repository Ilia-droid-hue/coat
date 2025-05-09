# coating_simulator_project/coating_simulator/gui/frames/emission.py
"""
Frame for configuring particle emission parameters.
Фрейм для настройки параметров эмиссии частиц.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import traceback
import math # Добавим импорт math для isclose

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

        Args:
            master: Parent widget. Родительский виджет.
            processing_frame_update_callback: Callback to notify processing frame of changes.
                                             Обратный вызов для уведомления фрейма обработки об изменениях.
            **kwargs: Additional arguments for ttk.LabelFrame.
                      Дополнительные аргументы для ttk.LabelFrame.
        """
        super().__init__(master, text="Параметры эмиссии", **kwargs)
        self.columnconfigure(1, weight=1) # Allow entry column to expand

        self._processing_update_callback = processing_frame_update_callback

        # Validation commands
        vcmd_float = (self.register(self._validate_float), '%P')
        vcmd_int = (self.register(self._validate_int), '%P')
        # Validation for full angle (0 <= angle <= 180) - ИЗМЕНЕНО
        vcmd_angle = (self.register(self._validate_full_angle), '%P')

        # --- Widgets ---
        # Distribution Type
        ttk.Label(self, text="Тип распределения:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5), pady=2)
        self.combo_dist = ttk.Combobox(self, values=config.DISTRIBUTION_TYPES, state="readonly", width=20) # Adjusted width
        self.combo_dist.set(config.DIST_GAUSSIAN) # Default
        self.combo_dist.grid(row=0, column=1, sticky=tk.EW, pady=2)

        # Full Emission Angle (Theta_full)
        ttk.Label(self, text="Полный угол эмиссии (°):").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=2)
        self.entry_full_angle = ttk.Entry(self, validate='key', validatecommand=vcmd_angle)
        # Устанавливаем значение по умолчанию как ПОЛНЫЙ угол (2 * max_theta)
        self.entry_full_angle.insert(0, str(config.DEFAULT_EMISSION_PARAMS["max_theta"] * 2.0))
        self.entry_full_angle.grid(row=1, column=1, sticky=tk.EW, pady=2)
        # --------------------------------------------------------

        # Number of Particles (Initially hidden, shown by ProcessingFrame)
        self.label_parts = ttk.Label(self, text="Частиц:")
        self.entry_parts = ttk.Entry(self, validate='key', validatecommand=vcmd_int)
        self.entry_parts.insert(0, str(config.DEFAULT_EMISSION_PARAMS["particles"]))

        # --- Distribution Specific Widgets (created but not gridded) ---
        self.specific_widgets = {}

        # Gaussian Beam
        self.label_sigma = ttk.Label(self, text="Sigma (σ), °:")
        self.entry_sigma = ttk.Entry(self, validate='key', validatecommand=vcmd_float)
        self.entry_sigma.insert(0, str(config.DEFAULT_EMISSION_PARAMS["sigma"]))
        self.specific_widgets[config.DIST_GAUSSIAN] = [
            (self.label_sigma, self.entry_sigma)
        ]

        # Cosine-power
        self.label_m = ttk.Label(self, text="Экспонента m:")
        self.entry_m = ttk.Entry(self, validate='key', validatecommand=vcmd_float)
        self.entry_m.insert(0, str(config.DEFAULT_EMISSION_PARAMS["m_exp"]))
        self.specific_widgets[config.DIST_COSINE_POWER] = [
            (self.label_m, self.entry_m)
        ]

        # Uniform Solid Angle - no specific widgets
        self.specific_widgets[config.DIST_UNIFORM_SOLID] = []

        # --- Bindings ---
        self.combo_dist.bind("<<ComboboxSelected>>", self._update_fields)

        # --- Initial State ---
        self._update_fields()

    def _validate_float(self, P):
        """Validation function: Allow empty string or valid float."""
        if P == "" or P == "-":
            return True
        # Allow '.' for starting decimal entry, but only one
        if P == ".": return True
        if P.count('.') > 1: return False
        try:
            float(P)
            return True
        except ValueError:
            return False

    def _validate_int(self, P):
        """Validation function: Allow empty string or valid integer."""
        if P == "":
            return True
        try:
            int(P)
            return True
        except ValueError:
            return False

    def _validate_full_angle(self, P):
        """Validation function: Allow empty or float between 0 (inclusive) and 180 (inclusive)."""
        if P == "":
            return True
        if P == ".": return True
        if P.count('.') > 1: return False
        try:
            val = float(P)
            # Угол должен быть >= 0 и <= 180 - ИЗМЕНЕНО
            return 0 <= val <= 180
        except ValueError:
            return False


    def _update_fields(self, event=None):
        """Shows/hides specific parameter fields based on distribution type."""
        selected_dist = self.combo_dist.get()

        # Hide all specific widgets first
        for dist_type in self.specific_widgets:
            for label, entry in self.specific_widgets[dist_type]:
                label.grid_remove()
                entry.grid_remove()

        # Show widgets for the selected type
        if selected_dist in self.specific_widgets:
            row_index = 3 # Start after angle and particles (if shown)
            for label, entry in self.specific_widgets[selected_dist]:
                label.grid(row=row_index, column=0, sticky=tk.W, padx=(0, 5), pady=2)
                entry.grid(row=row_index, column=1, sticky=tk.EW, pady=2)
                row_index += 1
        else:
            print(f"Warning: No specific fields defined for distribution type '{selected_dist}'")

        # Trigger processing frame update if callback exists
        if self._processing_update_callback:
             try:
                 self._processing_update_callback()
             except Exception:
                 print("Error in processing update callback from emission frame:")
                 traceback.print_exc()

    def get_params(self) -> dict:
        """
        Retrieves the configured emission parameters as a dictionary.
        Converts the input full angle to the half-angle (max_theta) used internally.
        Извлекает настроенные параметры эмиссии в виде словаря.
        Конвертирует введенный полный угол в половинный угол (max_theta), используемый внутри.

        Returns:
            A dictionary containing emission parameters.
            Словарь, содержащий параметры эмиссии.

        Raises:
            ValueError: If a required numeric parameter has invalid input.
                        Если требуемый числовой параметр имеет неверный ввод.
        """
        params = {}
        dist_type = self.combo_dist.get()
        params['dist_type'] = dist_type

        try:
            # Get full angle and convert to half-angle (max_theta)
            full_angle_deg = float(self.entry_full_angle.get())
            if not (0 <= full_angle_deg <= 180): # Re-check range just in case
                 raise ValueError("Полный угол эмиссии должен быть между 0 и 180 градусами.")
            params['max_theta'] = full_angle_deg / 2.0 # Store half-angle

            # Get number of particles
            params['particles'] = int(self.entry_parts.get())
            if params['particles'] <= 0:
                 raise ValueError("Число частиц должно быть положительным целым числом.")

            # Specific parameters
            if dist_type == config.DIST_GAUSSIAN:
                params['sigma'] = float(self.entry_sigma.get())
                if params['sigma'] <= 0:
                     raise ValueError("Sigma (σ) для Gaussian beam должна быть положительной.")
                 # Adjust warning: only warn if max_theta > 0 and sigma > max_theta
                 # Корректируем предупреждение: предупреждаем только если max_theta > 0 и sigma > max_theta
                if params['max_theta'] > 1e-6 and params['sigma'] > params['max_theta']:
                     print(f"Предупреждение: Sigma ({params['sigma']}°) больше половины полного угла ({params['max_theta']}°).")
            elif dist_type == config.DIST_COSINE_POWER:
                params['m_exp'] = float(self.entry_m.get())
                if params['m_exp'] < 0:
                     raise ValueError("Экспонента m для Cosine-power не может быть отрицательной.")

        except ValueError as e:
            messagebox.showerror("Ошибка ввода (Эмиссия)", f"Некорректное значение в параметрах эмиссии: {e}", parent=self)
            raise ValueError(f"Invalid emission parameter: {e}") from e
        except Exception as e:
             messagebox.showerror("Ошибка (Эмиссия)", f"Неожиданная ошибка при чтении параметров эмиссии: {e}", parent=self)
             raise RuntimeError(f"Unexpected error getting emission params: {e}") from e

        return params

    def show_particles_entry(self, row: int):
        """Grids the particle count entry at the specified row."""
        self.label_parts.grid(row=row, column=0, sticky=tk.W, padx=(0, 5), pady=2)
        self.entry_parts.grid(row=row, column=1, sticky=tk.EW, pady=2)

    def hide_particles_entry(self):
        """Removes the particle count entry from the grid."""
        self.label_parts.grid_remove()
        self.entry_parts.grid_remove()

