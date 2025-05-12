# coating_simulator_project/coating_simulator/gui/frames/source.py
"""
Frame for selecting source type and configuring its parameters.
Фрейм для выбора типа источника и настройки его параметров.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import traceback
import math

# Используем относительные импорты
from ... import config # Импортируем константы и хелперы

class SourceFrame(ttk.LabelFrame):
    """
    LabelFrame containing widgets for source configuration.
    LabelFrame, содержащий виджеты для конфигурации источника.
    """
    def __init__(self, master, **kwargs):
        """
        Initializes the SourceFrame.
        Инициализирует SourceFrame.

        Args:
            master: Parent widget. Родительский виджет.
            **kwargs: Additional arguments for ttk.LabelFrame.
                      Дополнительные аргументы для ttk.LabelFrame.
        """
        super().__init__(master, text="Параметры источника", **kwargs)
        self.columnconfigure(1, weight=1) # Allow entry column to expand

        # Validation command for numeric entries (allow float or empty)
        vcmd_float = (self.register(self._validate_float), '%P')
        # Validation for focus point (allow float, empty, or infinity symbol)
        vcmd_focus = (self.register(self._validate_focus), '%P')

        # --- Widgets ---
        # Source Type
        ttk.Label(self, text="Тип источника:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5), pady=2)
        self.combo_src = ttk.Combobox(self, values=config.SOURCE_TYPES, state="readonly", width=20)
        self.combo_src.set(config.SOURCE_POINT) # Default
        self.combo_src.grid(row=0, column=1, sticky=tk.EW, pady=2)

        # Common Parameters (Position and Rotation)
        ttk.Label(self, text="Смещение X (мм):").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=2)
        self.entry_srcx = ttk.Entry(self, validate='key', validatecommand=vcmd_float)
        self.entry_srcx.insert(0, str(config.DEFAULT_SOURCE_PARAMS["src_x"]))
        self.entry_srcx.grid(row=1, column=1, sticky=tk.EW, pady=2)

        ttk.Label(self, text="Смещение Y (мм):").grid(row=2, column=0, sticky=tk.W, padx=(0, 5), pady=2)
        self.entry_srcy = ttk.Entry(self, validate='key', validatecommand=vcmd_float)
        self.entry_srcy.insert(0, str(config.DEFAULT_SOURCE_PARAMS["src_y"]))
        self.entry_srcy.grid(row=2, column=1, sticky=tk.EW, pady=2)

        ttk.Label(self, text="Высота Z (мм):").grid(row=3, column=0, sticky=tk.W, padx=(0, 5), pady=2)
        self.entry_srcz = ttk.Entry(self, validate='key', validatecommand=vcmd_float)
        self.entry_srcz.insert(0, str(config.DEFAULT_SOURCE_PARAMS["src_z"]))
        self.entry_srcz.grid(row=3, column=1, sticky=tk.EW, pady=2)

        ttk.Label(self, text="Наклон вокруг X (°):").grid(row=4, column=0, sticky=tk.W, padx=(0, 5), pady=2)
        self.entry_rotx = ttk.Entry(self, validate='key', validatecommand=vcmd_float)
        self.entry_rotx.insert(0, str(config.DEFAULT_SOURCE_PARAMS["rot_x"]))
        self.entry_rotx.grid(row=4, column=1, sticky=tk.EW, pady=2)

        ttk.Label(self, text="Наклон вокруг Y (°):").grid(row=5, column=0, sticky=tk.W, padx=(0, 5), pady=2)
        self.entry_roty = ttk.Entry(self, validate='key', validatecommand=vcmd_float)
        self.entry_roty.insert(0, str(config.DEFAULT_SOURCE_PARAMS["rot_y"]))
        self.entry_roty.grid(row=5, column=1, sticky=tk.EW, pady=2)

        # --- Source Type Specific Widgets (created but not gridded) ---
        self.specific_widgets = {} # Dictionary to hold groups of specific widgets

        # Ring Source Widgets
        self.label_sd_ring = ttk.Label(self, text="Диаметр кольца (мм):")
        self.entry_sd_ring = ttk.Entry(self, validate='key', validatecommand=vcmd_float)
        self.entry_sd_ring.insert(0, str(config.DEFAULT_SOURCE_PARAMS["src_diameter"]))

        self.label_cone = ttk.Label(self, text="Угол конуса φ (°):")
        self.entry_cone = ttk.Entry(self, validate='key', validatecommand=vcmd_float)
        self.entry_cone.insert(0, str(config.DEFAULT_SOURCE_PARAMS["cone_angle"]))

        self.label_focus = ttk.Label(self, text="Точка фокуса L (мм):")
        self.entry_focus = ttk.Entry(self, validate='key', validatecommand=vcmd_focus)
        # Calculate initial focus point based on defaults
        initial_focus = config.calculate_focus_point(
            config.DEFAULT_SOURCE_PARAMS["src_diameter"],
            config.DEFAULT_SOURCE_PARAMS["cone_angle"]
        )
        self.entry_focus.insert(0, initial_focus)

        self.specific_widgets[config.SOURCE_RING] = [
            (self.label_sd_ring, self.entry_sd_ring),
            (self.label_cone, self.entry_cone),
            (self.label_focus, self.entry_focus),
        ]

        # Circular Source Widgets
        self.label_sd_circ = ttk.Label(self, text="Диаметр круга (мм):") # Different label text
                                                                          # Другой текст метки
        self.entry_sd_circ = ttk.Entry(self, validate='key', validatecommand=vcmd_float)
        self.entry_sd_circ.insert(0, str(config.DEFAULT_SOURCE_PARAMS["src_diameter"]))
        self.specific_widgets[config.SOURCE_CIRCULAR] = [
            (self.label_sd_circ, self.entry_sd_circ)
        ]

        # Linear Source Widgets
        self.label_sl = ttk.Label(self, text="Длина (мм):")
        self.entry_sl = ttk.Entry(self, validate='key', validatecommand=vcmd_float)
        self.entry_sl.insert(0, str(config.DEFAULT_SOURCE_PARAMS["src_length"]))

        self.label_sa = ttk.Label(self, text="Угол (°):")
        self.entry_sa = ttk.Entry(self, validate='key', validatecommand=vcmd_float)
        self.entry_sa.insert(0, str(config.DEFAULT_SOURCE_PARAMS["src_angle"]))
        self.specific_widgets[config.SOURCE_LINEAR] = [
            (self.label_sl, self.entry_sl),
            (self.label_sa, self.entry_sa),
        ]

        # Point source has no specific widgets
        self.specific_widgets[config.SOURCE_POINT] = []

        # --- Bindings ---
        self.combo_src.bind("<<ComboboxSelected>>", self._update_fields)
        # Bindings for ring source calculations
        self.entry_sd_ring.bind("<KeyRelease>", self._update_focus_from_angle_or_diam)
        self.entry_cone.bind("<KeyRelease>", self._update_focus_from_angle_or_diam)
        self.entry_focus.bind("<KeyRelease>", self._update_cone_from_focus)

        # --- Initial State ---
        self._update_fields()

    def _validate_float(self, P):
        """Validation function: Allow empty string or valid float."""
        if P == "" or P == "-": # Allow minus sign temporarily
                                # Временно разрешаем знак минуса
            return True
        try:
            float(P)
            return True
        except ValueError:
            return False

    def _validate_focus(self, P):
        """Validation function: Allow empty, float, or infinity symbol."""
        if P == "" or P == config.INFINITY_SYMBOL or P == "-":
            return True
        try:
            float(P)
            return True
        except ValueError:
            return False

    def _update_focus_from_angle_or_diam(self, event=None):
        """Recalculates Focus Point L when Diameter D or Cone Angle phi changes."""
        try:
            D_str = self.entry_sd_ring.get()
            phi_str = self.entry_cone.get()
            if not D_str or not phi_str: # Don't calculate if inputs are empty
                                         # Не вычисляем, если поля ввода пусты
                 return
            D = float(D_str)
            phi = float(phi_str)
            focus_val = config.calculate_focus_point(D, phi)

            # Update entry only if value changed to avoid cursor jumps/infinite loops
            # Обновляем поле ввода только если значение изменилось, чтобы избежать скачков курсора/бесконечных циклов
            if self.entry_focus.get() != focus_val:
                # Temporarily unbind to prevent recursive call from focus update
                # Временно отвязываем, чтобы предотвратить рекурсивный вызов из обновления фокуса
                self.entry_focus.unbind("<KeyRelease>")
                current_pos = self.entry_focus.index(tk.INSERT) # Store cursor position
                                                                 # Сохраняем позицию курсора
                self.entry_focus.delete(0, tk.END)
                self.entry_focus.insert(0, focus_val)
                self.entry_focus.icursor(current_pos) # Restore cursor position
                                                      # Восстанавливаем позицию курсора
                # Re-bind after a short delay
                # Повторно привязываем после небольшой задержки
                self.after(50, lambda: self.entry_focus.bind("<KeyRelease>", self._update_cone_from_focus))

        except ValueError:
            pass # Ignore errors during typing invalid numbers
                 # Игнорируем ошибки при вводе неверных чисел
        except Exception:
             traceback.print_exc() # Log other unexpected errors
                                   # Логируем другие неожиданные ошибки

    def _update_cone_from_focus(self, event=None):
        """Recalculates Cone Angle phi when Focus Point L or Diameter D changes."""
        try:
            D_str = self.entry_sd_ring.get()
            L_str = self.entry_focus.get()
            if not D_str or not L_str: # Don't calculate if inputs are empty
                return
            D = float(D_str)
            # L can be infinity symbol
            # L может быть символом бесконечности
            cone_val = config.calculate_cone_angle(D, L_str)

            if self.entry_cone.get() != cone_val:
                # Temporarily unbind to prevent recursive call
                self.entry_cone.unbind("<KeyRelease>")
                current_pos = self.entry_cone.index(tk.INSERT)
                self.entry_cone.delete(0, tk.END)
                self.entry_cone.insert(0, cone_val)
                self.entry_cone.icursor(current_pos)
                 # Re-bind after a short delay
                self.after(50, lambda: self.entry_cone.bind("<KeyRelease>", self._update_focus_from_angle_or_diam))

        except ValueError:
             # Handle case where L is not a valid float or infinity
             # Обрабатываем случай, когда L не является допустимым float или бесконечностью
             if L_str != config.INFINITY_SYMBOL:
                 pass # Ignore errors during typing invalid numbers
        except Exception:
             traceback.print_exc()

    def _update_fields(self, event=None):
        """Shows/hides specific parameter fields based on selected source type."""
        selected_type = self.combo_src.get()

        # Hide all specific widgets first
        for src_type in self.specific_widgets:
            for label, entry in self.specific_widgets[src_type]:
                label.grid_remove()
                entry.grid_remove()

        # Show widgets for the selected type
        if selected_type in self.specific_widgets:
            # Start gridding after the common parameters (row 6)
            # Начинаем размещение после общих параметров (строка 6)
            row_index = 6
            for label, entry in self.specific_widgets[selected_type]:
                label.grid(row=row_index, column=0, sticky=tk.W, padx=(0, 5), pady=2)
                entry.grid(row=row_index, column=1, sticky=tk.EW, pady=2)
                row_index += 1
        else:
             print(f"Warning: No specific fields defined for source type '{selected_type}'")

    def get_params(self) -> dict:
        """
        Retrieves the configured source parameters as a dictionary.
        Извлекает настроенные параметры источника в виде словаря.

        Returns:
            A dictionary containing source parameters.
            Словарь, содержащий параметры источника.

        Raises:
            ValueError: If a required numeric parameter has invalid input.
                        Если требуемый числовой параметр имеет неверный ввод.
        """
        params = {}
        source_type = self.combo_src.get()
        params['src_type'] = source_type

        try:
            # Common parameters
            params['src_x'] = float(self.entry_srcx.get())
            params['src_y'] = float(self.entry_srcy.get())
            params['src_z'] = float(self.entry_srcz.get())
            params['rot_x'] = float(self.entry_rotx.get())
            params['rot_y'] = float(self.entry_roty.get())

            # Specific parameters
            if source_type == config.SOURCE_RING:
                params['src_diameter'] = float(self.entry_sd_ring.get())
                params['cone_angle'] = float(self.entry_cone.get())
                focus_raw = self.entry_focus.get()
                # Store focus as is (string '∞' or float string) - simulation core will handle it
                # Сохраняем фокус как есть (строка '∞' или строка float) - ядро симуляции обработает это
                params['focus_point'] = focus_raw if focus_raw == config.INFINITY_SYMBOL else float(focus_raw)
                if params['src_diameter'] <= 0: raise ValueError("Диаметр кольца должен быть положительным.")
            elif source_type == config.SOURCE_CIRCULAR:
                params['src_diameter'] = float(self.entry_sd_circ.get())
                if params['src_diameter'] <= 0: raise ValueError("Диаметр круга должен быть положительным.")
            elif source_type == config.SOURCE_LINEAR:
                params['src_length'] = float(self.entry_sl.get())
                params['src_angle'] = float(self.entry_sa.get())
                if params['src_length'] <= 0: raise ValueError("Длина линейного источника должна быть положительной.")
            # Point source: no specific params needed beyond common ones

            # Validate Z > 0 (usually required)
            if params['src_z'] <= 0:
                 raise ValueError("Высота источника Z должна быть положительной.")

        except ValueError as e:
            messagebox.showerror("Ошибка ввода (Источник)", f"Некорректное значение в параметрах источника: {e}", parent=self)
            raise ValueError(f"Invalid source parameter: {e}") from e
        except Exception as e:
             messagebox.showerror("Ошибка (Источник)", f"Неожиданная ошибка при чтении параметров источника: {e}", parent=self)
             raise RuntimeError(f"Unexpected error getting source params: {e}") from e

        return params

# Example usage (for testing purposes)
if __name__ == '__main__':
    root = tk.Tk()
    root.title("Тест SourceFrame")

    frame = SourceFrame(root, padding=10)
    frame.pack(expand=True, fill=tk.BOTH)

    def print_params():
        try:
            params = frame.get_params()
            print("Полученные параметры:", params)
        except (ValueError, RuntimeError) as e:
            print("Ошибка получения параметров:", e)

    button = ttk.Button(root, text="Получить параметры", command=print_params)
    button.pack(pady=10)

    root.mainloop()
