# coating_simulator_project/run_gui.py
"""
Entry point script to launch the Coating Simulator GUI application.
Скрипт точки входа для запуска GUI-приложения Симулятора Покрытия.

Run this file to start the application.
Запустите этот файл, чтобы начать работу приложения.
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox, scrolledtext
import traceback # Import traceback for detailed error printing
import multiprocessing # <--- ДОБАВЛЕН ИМПОРТ

# --- Add project root to Python path ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ---------------------------------------

# print("--- sys.path ---")
# for p in sys.path:
#     print(p)
# print("----------------")
# print(f"Project root added to path: {project_root}")
# print(f"Current working directory: {os.getcwd()}")

# --- Step-by-step Import Test ---
try:
    # print("\n--- Import Test ---")

    # print("1. Attempting: import coating_simulator")
    import coating_simulator
    # print("   Success: import coating_simulator")
    # print(f"   Location: {coating_simulator.__file__}")

    # print("\n2. Attempting: from coating_simulator import config")
    from coating_simulator import config
    # print("   Success: from coating_simulator import config")

    # print("\n3. Attempting: from coating_simulator import gui")
    from coating_simulator import gui
    # print("   Success: from coating_simulator import gui")
    # print(f"   Location: {gui.__file__}")

    # print("\n4. Attempting: from coating_simulator.gui import app")
    from coating_simulator.gui import app
    # print("   Success: from coating_simulator.gui import app")
    # print(f"   Location: {app.__file__}")

    # print("\n5. Attempting: from coating_simulator.gui.app import App")
    from coating_simulator.gui.app import App
    # print("   Success: from coating_simulator.gui.app import App")
    # print("--- Import Test Complete ---")

except ImportError as e:
    print(f"\n--- ImportError Occurred ---", file=sys.stderr)
    print(f"Error during import step: {e}", file=sys.stderr)
    print(f"Please double-check the existence and content (should be empty) of all '__init__.py' files", file=sys.stderr)
    print(f"Affected path might be: {e.path}" if hasattr(e, 'path') else "", file=sys.stderr)
    traceback.print_exc()
    try:
        root_err = tk.Tk()
        root_err.withdraw()
        messagebox.showerror("Ошибка Запуска (Импорт)",
                             "Не удалось импортировать компоненты приложения.\n"
                             f"Проверьте __init__.py и имена папок.\n\nДетали: {e}")
        root_err.destroy()
    except tk.TclError:
        pass
    sys.exit(1)
except Exception as e:
    print(f"\n--- An unexpected error occurred during import test ---", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)


# --- Dependency Check and App Launch (only if imports succeed) ---
if __name__ == "__main__":
    # !!! ВАЖНО для multiprocessing на Windows и при сборке в exe !!!
    multiprocessing.freeze_support() # <--- ДОБАВЛЕНО

    # print("\n--- Dependency Check ---") # Можно закомментировать для более чистого вывода
    try:
        import numpy
        # print("   Found numpy")
        import matplotlib
        # print("   Found matplotlib")
        # print("--- Dependencies OK ---")
    except ImportError as dep_error:
        print(f"Ошибка: Отсутствует необходимая библиотека - {dep_error.name}.", file=sys.stderr)
        print("Пожалуйста, установите ее, например: pip install numpy matplotlib", file=sys.stderr)
        try:
            root_dep_err = tk.Tk()
            root_dep_err.withdraw()
            messagebox.showerror("Ошибка Зависимостей",
                                 f"Отсутствует необходимая библиотека: {dep_error.name}.\n"
                                 "Пожалуйста, установите ее (например, через pip)\n"
                                 "и попробуйте запустить приложение снова.")
            root_dep_err.destroy()
        except tk.TclError:
            pass
        sys.exit(1)

    # print("\n--- Launching Application ---") # Можно закомментировать
    app_instance = App()
    app_instance.protocol("WM_DELETE_WINDOW", app_instance.on_closing)
    app_instance.mainloop()
