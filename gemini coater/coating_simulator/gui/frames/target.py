# coating_simulator_project/coating_simulator/gui/frames/target.py
"""
Frame for selecting target type and configuring its parameters.
Фрейм для выбора типа мишени и настройки ее параметров.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import traceback # For detailed error logging

# Используем относительные импорты
from ... import config # Импортируем константы

class TargetFrame(ttk.LabelFrame):
    """
    LabelFrame containing widgets for target configuration.
    LabelFrame, содержащий виджеты для конфигурации мишени.
    """
    def __init__(self, master, target_update_callback=None, **kwargs):
        """
        Initializes the TargetFrame.
        Инициализирует TargetFrame.

        Args:
            master: Parent widget. Родительский виджет.
            target_update_callback: Function to call when target type changes.
                                    Функция, вызываемая при изменении типа мишени.
            **kwargs: Additional arguments for ttk.LabelFrame.
                      Дополнительные аргументы для ttk.LabelFrame.
        """
        super().__init__(master, text="Параметры мишени", **kwargs)
        self.columnconfigure(1, weight=1) # Allow entry column to expand
                                          # Разрешаем колонке с полями ввода расширяться

        self._target_update_callback = target_update_callback

        # Validation command for numeric entries (allow float or empty)
        # Команда валидации для числовых полей (разрешает float или пустое значение)
        vcmd = (self.register(self._validate_float), '%P') # '%P' passes the potential value

        # --- Widgets ---
        # Target Type Combobox
        # Комбобокс типа мишени
        ttk.Label(self, text="Тип мишени:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.combo_tgt = ttk.Combobox(self, values=config.TARGET_TYPES, state="readonly", width=20) # Increased width
                                                                                                    # Увеличена ширина
        self.combo_tgt.set(config.TARGET_DISK) # Default selection
                                               # Выбор по умолчанию
        self.combo_tgt.grid(row=0, column=1, sticky=tk.EW) # Use EW for expansion
                                                           # Используем EW для расширения

        # --- Parameter Entries (created but not gridded initially) ---
        # Поля ввода параметров (созданы, но изначально не размещены в сетке)
        self.widgets_map = {} # Dictionary to hold widgets for each target type
                              # Словарь для хранения виджетов для каждого типа мишени

        # Disk / Dome
        self.label_disk_diameter = ttk.Label(self, text="Диаметр (мм):")
        self.entry_disk_diameter = ttk.Entry(self, validate='key', validatecommand=vcmd)
        self.entry_disk_diameter.insert(0, str(config.DEFAULT_TARGET_PARAMS["diameter"]))
        self.widgets_map[config.TARGET_DISK] = [
            (self.label_disk_diameter, self.entry_disk_diameter)
        ]

        self.label_dome_radius = ttk.Label(self, text="Радиус купола (мм):")
        self.entry_dome_radius = ttk.Entry(self, validate='key', validatecommand=vcmd)
        self.entry_dome_radius.insert(0, str(config.DEFAULT_TARGET_PARAMS["dome_radius"]))
        self.widgets_map[config.TARGET_DOME] = [
            (self.label_disk_diameter, self.entry_disk_diameter), # Reuse diameter
                                                                  # Повторно используем диаметр
            (self.label_dome_radius, self.entry_dome_radius)
        ]

        # Linear Movement
        self.label_length = ttk.Label(self, text="Длина (мм):")
        self.entry_length = ttk.Entry(self, validate='key', validatecommand=vcmd)
        self.entry_length.insert(0, str(config.DEFAULT_TARGET_PARAMS["length"]))

        self.label_width = ttk.Label(self, text="Ширина (мм):")
        self.entry_width = ttk.Entry(self, validate='key', validatecommand=vcmd)
        self.entry_width.insert(0, str(config.DEFAULT_TARGET_PARAMS["width"]))
        self.widgets_map[config.TARGET_LINEAR] = [
            (self.label_length, self.entry_length),
            (self.label_width, self.entry_width)
        ]
        # Bind width entry for potential use in processing frame (e.g., passes calculation)
        # Привязываем поле ширины для потенциального использования во фрейме обработки (например, расчет проходов)
        self.entry_width.bind("<KeyRelease>", self._on_param_change)


        # Planetary
        self.label_orbit_diameter = ttk.Label(self, text="Диаметр орбиты (мм):")
        self.entry_orbit_diameter = ttk.Entry(self, validate='key', validatecommand=vcmd)
        self.entry_orbit_diameter.insert(0, str(config.DEFAULT_TARGET_PARAMS["orbit_diameter"]))

        self.label_planet_disk_diameter = ttk.Label(self, text="Диаметр диска планеты (мм):")
        self.entry_planet_disk_diameter = ttk.Entry(self, validate='key', validatecommand=vcmd)
        self.entry_planet_disk_diameter.insert(0, str(config.DEFAULT_TARGET_PARAMS["planet_disk_diameter"]))
        self.widgets_map[config.TARGET_PLANETARY] = [
            (self.label_orbit_diameter, self.entry_orbit_diameter),
            (self.label_planet_disk_diameter, self.entry_planet_disk_diameter)
        ]

        # --- Bindings and Initial State ---
        self.combo_tgt.bind("<<ComboboxSelected>>", self._update_fields)
        self._update_fields() # Show fields for the default selection
                              # Показываем поля для выбора по умолчанию

    def _validate_float(self, P):
        """Validation function: Allow empty string or valid float."""
        if P == "":
            return True
        try:
            float(P)
            return True
        except ValueError:
            return False

    def _on_param_change(self, event=None):
        """Callback when a parameter entry changes (used for inter-frame updates)."""
        if self._target_update_callback:
            try:
                # Pass the current target type and the specific widget that changed
                # Передаем текущий тип мишени и конкретный изменившийся виджет
                self._target_update_callback(self.combo_tgt.get(), event.widget if event else None)
            except Exception:
                 # Log error but don't crash GUI
                 # Логируем ошибку, но не приводим к сбою GUI
                 print("Error in target update callback:")
                 traceback.print_exc()


    def _update_fields(self, event=None):
        """Shows/hides parameter entry fields based on selected target type."""
        selected_type = self.combo_tgt.get()

        # Hide all parameter widgets first
        # Сначала скрываем все виджеты параметров
        for target_type in self.widgets_map:
            for label, entry in self.widgets_map[target_type]:
                label.grid_remove()
                entry.grid_remove()

        # Show widgets for the selected type
        # Показываем виджеты для выбранного типа
        if selected_type in self.widgets_map:
            row_index = 1 # Start gridding from row 1
                          # Начинаем размещение с 1-й строки
            for label, entry in self.widgets_map[selected_type]:
                label.grid(row=row_index, column=0, sticky=tk.W, padx=(0, 5), pady=2)
                entry.grid(row=row_index, column=1, sticky=tk.EW, pady=2)
                row_index += 1
        else:
            print(f"Warning: No fields defined for target type '{selected_type}'")

        # Notify other parts of the GUI that the target type might have changed
        # Уведомляем другие части GUI об изменении типа мишени
        if self._target_update_callback:
             try:
                 self._target_update_callback(selected_type, self.combo_tgt) # Pass combobox as source
                                                                             # Передаем комбобокс как источник
             except Exception:
                 print("Error in target update callback:")
                 traceback.print_exc()


    def get_params(self) -> dict:
        """
        Retrieves the configured target parameters as a dictionary.
        Извлекает настроенные параметры мишени в виде словаря.

        Returns:
            A dictionary containing target parameters.
            Словарь, содержащий параметры мишени.

        Raises:
            ValueError: If a required numeric parameter has invalid input.
                        Если требуемый числовой параметр имеет неверный ввод.
        """
        params = {}
        target_type = self.combo_tgt.get()
        params['target_type'] = target_type

        try:
            if target_type == config.TARGET_DISK:
                params['diameter'] = float(self.entry_disk_diameter.get())
            elif target_type == config.TARGET_DOME:
                params['diameter'] = float(self.entry_disk_diameter.get())
                params['dome_radius'] = float(self.entry_dome_radius.get())
            elif target_type == config.TARGET_LINEAR:
                params['length'] = float(self.entry_length.get())
                params['width'] = float(self.entry_width.get())
            elif target_type == config.TARGET_PLANETARY:
                params['orbit_diameter'] = float(self.entry_orbit_diameter.get())
                params['planet_diameter'] = float(self.entry_planet_disk_diameter.get())
            else:
                 # Should not happen with combobox, but good practice
                 # Не должно происходить с комбобоксом, но хорошая практика
                raise ValueError(f"Неизвестный тип мишени: {target_type}")

            # Basic validation for positive dimensions
            # Базовая валидация положительных размеров
            for key, value in params.items():
                if isinstance(value, (float, int)) and value <= 0 and key != 'target_type':
                     # Find the corresponding entry widget to display error near it (optional)
                     # Находим соответствующий виджет Entry для отображения ошибки рядом с ним (опционально)
                     widget_name = ""
                     if key == 'diameter': widget_name = "Диаметр"
                     elif key == 'dome_radius': widget_name = "Радиус купола"
                     elif key == 'length': widget_name = "Длина"
                     elif key == 'width': widget_name = "Ширина"
                     elif key == 'orbit_diameter': widget_name = "Диаметр орбиты"
                     elif key == 'planet_diameter': widget_name = "Диаметр диска планеты"

                     raise ValueError(f"Параметр '{widget_name}' ({key}) должен быть положительным числом.")

        except ValueError as e:
            # Reraise with a more user-friendly message potentially
            # Повторно вызываем с потенциально более дружелюбным сообщением
            messagebox.showerror("Ошибка ввода (Мишень)", f"Некорректное значение в параметрах мишени: {e}", parent=self)
            raise ValueError(f"Invalid target parameter: {e}") from e
        except Exception as e:
             # Catch any other unexpected errors during parameter retrieval
             # Ловим любые другие неожиданные ошибки при получении параметров
             messagebox.showerror("Ошибка (Мишень)", f"Неожиданная ошибка при чтении параметров мишени: {e}", parent=self)
             raise RuntimeError(f"Unexpected error getting target params: {e}") from e


        return params

    def get_current_target_type(self) -> str:
        """Returns the currently selected target type."""
        return self.combo_tgt.get()

    def get_entry_widget(self, param_name: str) -> ttk.Entry | None:
         """ Returns the entry widget associated with a parameter name (e.g., 'width'). """
         if param_name == 'width' and hasattr(self, 'entry_width'):
             return self.entry_width
         # Add other mappings if needed for callbacks
         # Добавьте другие сопоставления, если необходимо для обратных вызовов
         return None


# Example usage (for testing purposes)
if __name__ == '__main__':
    root = tk.Tk()
    root.title("Тест TargetFrame")

    def on_target_update(target_type, widget):
        print(f"Target type updated to: {target_type}, triggered by: {widget}")
        # Example: Access width entry if needed
        width_entry = frame.get_entry_widget('width')
        if width_entry:
             print(f"  Current width value: {width_entry.get()}")


    frame = TargetFrame(root, target_update_callback=on_target_update, padding=10)
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
