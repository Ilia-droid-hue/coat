# coating_simulator_project/coating_simulator/gui/geometry_view_window.py
"""
Toplevel window for displaying the 2D geometry visualization using Matplotlib.
Features auto-scaled square plot areas with equal aspect ratio, toolbar, and legend below.
Дочернее окно Toplevel для отображения 2D-визуализации геометрии с помощью Matplotlib.
Имеет автомасштабируемые квадратные области графиков с равным соотношением сторон, панель инструментов и легенду внизу.
"""
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import traceback

# Используем относительные импорты
from ..visualization import geom_layout # Импортируем наш модуль с основной функцией
from .. import config


class GeometryViewWindow(tk.Toplevel):
    """
    A Toplevel window that embeds Matplotlib plots for 2D geometry visualization.
    Дочернее окно Toplevel, в которое встроены графики Matplotlib для 2D-визуализации геометрии.
    """
    def __init__(self, parent, params: dict):
        """
        Initializes the GeometryViewWindow.
        Инициализирует GeometryViewWindow.

        Args:
            parent: The parent widget (main application window).
            params (dict): Dictionary containing simulation parameters needed for plotting.
        """
        super().__init__(parent)
        self.title("Предпросмотр геометрии (2D)")
        self.minsize(650, 500) # Минимальный размер, чтобы все поместилось

        self._params = params
        self._parent = parent

        # --- Create Matplotlib Figure ---
        # Размер фигуры влияет на размер окна по умолчанию, можно поэкспериментировать
        self.figure = Figure(figsize=(9, 5), dpi=100) # Уменьшили высоту фигуры

        # --- Frame to hold Canvas and Toolbar ---
        # Тулбар будет под канвасом
        toolbar_frame = ttk.Frame(self)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 2)) # Тулбар внизу

        canvas_frame = ttk.Frame(self) # Фрейм для канваса над тулбаром
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


        # --- Embed Matplotlib Figure in Tkinter Window ---
        self.canvas = FigureCanvasTkAgg(self.figure, master=canvas_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- Add Navigation Toolbar ---
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()


        # --- Plot the geometry and get legend info ---
        self.legend_handles = []
        self.legend_labels = []
        self._plot_geometry() # This calls plot_geometry_2d

        # --- Add Legend Below ---
        if self.legend_handles:
             num_labels = len(self.legend_labels)
             # Подбираем количество колонок для легенды
             ncol = min(max(1, num_labels // 2 if num_labels > 4 else num_labels), 4)

             # Создаем легенду под фигурой
             self.figure.legend(self.legend_handles, self.legend_labels,
                                loc='lower center',
                                bbox_to_anchor=(0.5, 0.01), # Располагаем чуть выше нижнего края фигуры
                                ncol=ncol,
                                fontsize='x-small', # Уменьшим шрифт легенды
                                bbox_transform=self.figure.transFigure)

             # Корректируем отступы для графиков, чтобы освободить место для легенды
             # Эти значения могут потребовать тонкой настройки
             bottom_margin = 0.12 + (num_labels // (ncol * 10)) * 0.04
             self.figure.subplots_adjust(left=0.07, right=0.97, bottom=max(0.15, bottom_margin), top=0.93, wspace=0.2)
             self.canvas.draw()


        # --- Center window ---
        self._center_window()

        # --- Handle closing ---
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _plot_geometry(self):
        """Calls the 2D plotting function and stores legend info."""
        try:
            handles, labels = geom_layout.plot_geometry_2d(self.figure, self._params)
            self.legend_handles = handles
            self.legend_labels = labels
            self.canvas.draw()
        except Exception as e:
            error_message = f"Не удалось построить 2D геометрию:\n{e}"
            print(error_message)
            traceback.print_exc()
            messagebox.showerror("Ошибка Визуализации Геометрии", error_message, parent=self)
            self.destroy()

    def _center_window(self):
        """Centers the window on its parent."""
        self.update_idletasks()
        parent_x = self._parent.winfo_x(); parent_y = self._parent.winfo_y()
        parent_width = self._parent.winfo_width(); parent_height = self._parent.winfo_height()
        # Get actual window size after packing widgets
        window_width = self.winfo_reqwidth()
        window_height = self.winfo_reqheight()
        
        center_x = parent_x + (parent_width - window_width) // 2
        center_y = parent_y + (parent_height - window_height) // 2
        center_x = max(0, center_x); center_y = max(0, center_y)
        self.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")


    def _on_closing(self):
        """Handles window closing."""
        plt.close(self.figure)
        if hasattr(self._parent, 'geometry_window_instance'):
             self._parent.geometry_window_instance = None
        self.destroy()

# Example usage remains commented out
if __name__ == '__main__':
    print("Для тестирования этого окна запустите основной файл run_gui.py")

