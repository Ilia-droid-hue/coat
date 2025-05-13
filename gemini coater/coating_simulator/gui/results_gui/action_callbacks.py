# coding: utf-8
# Файл: coating_simulator/gui/results_gui/action_callbacks.py
"""
Содержит функции-заглушки для действий, вызываемых из ResultsWindow,
таких как экспорт, расчет маски, загрузка профилей и реконструкция карты.
"""
import tkinter as tk # Нужен для messagebox
from tkinter import messagebox, filedialog
import os # Для работы с путями файлов при загрузке

# Этот модуль пока не требует сложных импортов из проекта,
# так как содержит в основном заглушки.

def placeholder_calculate_mask(parent_window, auto_uniformity_params: dict | None):
    """
    Заглушка для функции расчета маски для авторавномерности.
    Отображает информационное сообщение.

    Args:
        parent_window: Родительское окно для messagebox.
        auto_uniformity_params (dict | None): Параметры для расчета,
                                              например {'mode': 'Маска', 'mask_height': 50.0}.
                                              Если None, значит параметры не были получены.
    """
    if auto_uniformity_params:
        mode = auto_uniformity_params.get('mode', 'N/A')
        height = auto_uniformity_params.get('mask_height', 'N/A')
        message = (f"Функция 'Расчет маски для авторавномерности' еще не реализована.\n\n"
                   f"Выбранный режим: {mode}\n"
                   f"Высота маски: {height} мм")
        if hasattr(parent_window, 'info_display_area') and \
           hasattr(parent_window.info_display_area, 'update_auto_uniformity_info'):
            parent_window.info_display_area.update_auto_uniformity_info(
                f"Расчет маски ({mode}, {height}мм)... (заглушка)"
            )
    else:
        message = "Не удалось получить параметры для расчета маски."
        if hasattr(parent_window, 'info_display_area') and \
           hasattr(parent_window.info_display_area, 'update_auto_uniformity_info'):
            parent_window.info_display_area.update_auto_uniformity_info(
                "Ошибка получения параметров для маски."
            )
    messagebox.showinfo("Заглушка", message, parent=parent_window)


def placeholder_export_excel(parent_window):
    """
    Заглушка для функции экспорта данных в Excel.
    Отображает информационное сообщение.

    Args:
        parent_window: Родительское окно для messagebox.
    """
    messagebox.showinfo("Заглушка", "Функция 'Экспорт данных в Excel' еще не реализована.", parent=parent_window)


def placeholder_load_profiles(parent_window):
    """
    Заглушка для функции загрузки профилей для обратной задачи.
    Имитирует диалог выбора файлов и обновляет текстовое поле в GUI.

    Args:
        parent_window: Родительское окно (ResultsWindow).
    """
    filepaths = filedialog.askopenfilenames(
        title="Выберите файлы профилей (.csv, .txt)",
        filetypes=[("CSV файлы", "*.csv"), ("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")],
        parent=parent_window
    )

    if filepaths:
        # Здесь должна быть логика загрузки и обработки данных из файлов.
        # Пока просто сохраняем имена файлов.
        parent_window.loaded_profiles_data = list(filepaths) # Сохраняем полные пути
        filenames = [os.path.basename(fp) for fp in filepaths]
        loaded_text = "Загружено: " + ", ".join(filenames) if filenames else "Файлы не загружены"
        info_text = f"Загружено файлов: {len(filepaths)}.\nГотово к реконструкции."

        # Обновляем GUI через родительское окно (ResultsWindow)
        if hasattr(parent_window, 'settings_panel') and \
           hasattr(parent_window.settings_panel, 'update_loaded_files_text'):
            parent_window.settings_panel.update_loaded_files_text(loaded_text)

        if hasattr(parent_window, 'info_display_area') and \
           hasattr(parent_window.info_display_area, 'update_inverse_problem_info'):
            parent_window.info_display_area.update_inverse_problem_info(info_text)

        messagebox.showinfo("Загрузка профилей (Заглушка)",
                            f"Выбрано файлов: {len(filepaths)}.\n"
                            f"Имена: {', '.join(filenames)}\n\n"
                            "Дальнейшая обработка и реконструкция еще не реализованы.",
                            parent=parent_window)
    else:
        if hasattr(parent_window, 'settings_panel') and \
           hasattr(parent_window.settings_panel, 'update_loaded_files_text'):
            parent_window.settings_panel.update_loaded_files_text("Загрузка отменена")

        if hasattr(parent_window, 'info_display_area') and \
           hasattr(parent_window.info_display_area, 'update_inverse_problem_info'):
            parent_window.info_display_area.update_inverse_problem_info("Файлы профилей не загружены.")


def placeholder_reconstruct_map(parent_window, reconstruction_method: str):
    """
    Заглушка для функции реконструкции карты покрытия по загруженным профилям.
    Отображает информационное сообщение.

    Args:
        parent_window: Родительское окно (ResultsWindow).
        reconstruction_method (str): Выбранный метод реконструкции.
    """
    # Проверяем, были ли загружены профили (через атрибут в ResultsWindow)
    if not hasattr(parent_window, 'loaded_profiles_data') or not parent_window.loaded_profiles_data:
        messagebox.showwarning("Обратная задача", "Сначала загрузите файлы профилей.", parent=parent_window)
        if hasattr(parent_window, 'info_display_area') and \
           hasattr(parent_window.info_display_area, 'update_inverse_problem_info'):
            parent_window.info_display_area.update_inverse_problem_info("Ошибка: Профили не загружены.")
        return

    message = (f"Функция 'Построение карты по профилям' еще не реализована.\n\n"
               f"Выбранный метод реконструкции: {reconstruction_method}\n"
               f"Количество загруженных профилей: {len(parent_window.loaded_profiles_data)}")

    if hasattr(parent_window, 'info_display_area') and \
       hasattr(parent_window.info_display_area, 'update_inverse_problem_info'):
        parent_window.info_display_area.update_inverse_problem_info(
            f"Реконструкция ({reconstruction_method})...\n(Функция не реализована)"
        )
    messagebox.showinfo("Заглушка", message, parent=parent_window)


if __name__ == '__main__':
    # Пример использования (не будет работать без родительского окна Tkinter)
    print("Тестирование action_callbacks.py")
    print("Этот модуль содержит функции-заглушки и предназначен для импорта.")
    # Для теста можно создать мок-окно, но это выходит за рамки простого примера.
    # root = tk.Tk()
    # root.withdraw() # Скрыть основное окно Tk
    # placeholder_export_excel(root)
    # placeholder_calculate_mask(root, {'mode': 'Маска', 'mask_height': 10.0})
    # root.destroy()
