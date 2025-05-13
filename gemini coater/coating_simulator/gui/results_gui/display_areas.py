# coding: utf-8
# Файл: coating_simulator/gui/results_gui/display_areas.py
"""
Содержит классы PlotDisplayArea и InfoDisplayArea для окна результатов.
Эти классы были выделены из ursprüngльного results_panels.py.
Скорректирована логика растягивания для лучшего отображения.
Добавлена опция 'uniform' для columnconfigure.
Добавлена обработка события <Configure> для принудительного tight_layout.
"""
import tkinter as tk
from tkinter import ttk
import traceback

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class PlotDisplayArea(ttk.Frame):
    """
    Фрейм для отображения графиков (карты покрытия и профиля).
    """
    def __init__(self, master, plot_size_pixels=300, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.plot_size_pixels = plot_size_pixels
        self._resize_timer_map = None # Таймер для отложенного ресайза карты
        self._resize_timer_profile = None # Таймер для отложенного ресайза профиля

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        self._create_widgets()

    def _create_widgets(self):
        gutter_size = 2

        plots_container = ttk.Frame(self)
        plots_container.grid(row=0, column=0, sticky=tk.NSEW, pady=(0, gutter_size))
        plots_container.columnconfigure(0, weight=1, uniform="plotgroup")
        plots_container.columnconfigure(1, weight=1, uniform="plotgroup")
        plots_container.rowconfigure(0, weight=1)

        plot_dpi = self.winfo_fpixels('1i')
        if not isinstance(plot_dpi, (float, int)) or plot_dpi <= 0:
            plot_dpi = 96.0
        plot_size_inches = self.plot_size_pixels / plot_dpi

        # --- Карта покрытия ---
        self.map_plot_container = ttk.Frame(plots_container) # Сохраняем как атрибут
        self.map_plot_container.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, gutter_size // 2))
        self.map_plot_container.columnconfigure(0, weight=1)
        self.map_plot_container.rowconfigure(0, weight=1)
        self.map_figure = Figure(figsize=(plot_size_inches, plot_size_inches), dpi=plot_dpi)
        self.map_canvas = FigureCanvasTkAgg(self.map_figure, master=self.map_plot_container)
        self.map_canvas_widget = self.map_canvas.get_tk_widget()
        self.map_canvas_widget.grid(row=0, column=0, sticky=tk.NSEW)
        # Привязка события Configure к контейнеру карты
        self.map_plot_container.bind("<Configure>", self._on_map_container_configure)

        # --- Профиль покрытия ---
        self.profile_plot_container = ttk.Frame(plots_container) # Сохраняем как атрибут
        self.profile_plot_container.grid(row=0, column=1, sticky=tk.NSEW, padx=(gutter_size // 2, 0))
        self.profile_plot_container.columnconfigure(0, weight=1)
        self.profile_plot_container.rowconfigure(0, weight=1)
        self.profile_figure = Figure(figsize=(plot_size_inches, plot_size_inches), dpi=plot_dpi)
        self.profile_canvas = FigureCanvasTkAgg(self.profile_figure, master=self.profile_plot_container)
        self.profile_canvas_widget = self.profile_canvas.get_tk_widget()
        self.profile_canvas_widget.grid(row=0, column=0, sticky=tk.NSEW)
        # Привязка события Configure к контейнеру профиля
        self.profile_plot_container.bind("<Configure>", self._on_profile_container_configure)

        # --- Панель инструментов ---
        common_toolbar_frame = ttk.Frame(self)
        common_toolbar_frame.grid(row=1, column=0, sticky=tk.EW)
        self.toolbar = NavigationToolbar2Tk(self.map_canvas, common_toolbar_frame)
        self.toolbar.update()

    def _on_map_container_configure(self, event=None):
        """Обработчик изменения конфигурации контейнера карты."""
        if self._resize_timer_map:
            self.after_cancel(self._resize_timer_map)
        self._resize_timer_map = self.after(50, self._apply_tight_layout_map) # 50 мс задержка

    def _on_profile_container_configure(self, event=None):
        """Обработчик изменения конфигурации контейнера профиля."""
        if self._resize_timer_profile:
            self.after_cancel(self._resize_timer_profile)
        self._resize_timer_profile = self.after(50, self._apply_tight_layout_profile)

    def _apply_tight_layout_map(self):
        """Применяет tight_layout к фигуре карты и перерисовывает."""
        try:
            if self.map_figure and self.map_canvas:
                self.map_figure.tight_layout(pad=1.0)
                self.map_canvas.draw_idle()
        except Exception as e:
            print(f"Ошибка в _apply_tight_layout_map: {e}")
            # traceback.print_exc() # Раскомментировать для детальной отладки
        finally:
            self._resize_timer_map = None

    def _apply_tight_layout_profile(self):
        """Применяет tight_layout к фигуре профиля и перерисовывает."""
        try:
            if self.profile_figure and self.profile_canvas:
                self.profile_figure.tight_layout(pad=0.8)
                self.profile_canvas.draw_idle()
        except Exception as e:
            print(f"Ошибка в _apply_tight_layout_profile: {e}")
            # traceback.print_exc()
        finally:
            self._resize_timer_profile = None

    def get_map_figure(self) -> Figure:
        return self.map_figure

    def get_profile_figure(self) -> Figure:
        return self.profile_figure

    def draw_canvases(self):
        """Перерисовывает оба холста и применяет tight_layout."""
        # Первичная отрисовка может быть без tight_layout,
        # т.к. <Configure> обработчики вызовут его позже.
        # Однако, для согласованности, можно вызвать и здесь.
        try:
            if self.map_figure and self.map_canvas:
                self._apply_tight_layout_map() # Применяем сразу при явном вызове draw_canvases
            if self.profile_figure and self.profile_canvas:
                self._apply_tight_layout_profile()
        except Exception as e:
            print(f"Ошибка при вызове draw_canvases: {e}"); traceback.print_exc()


class InfoDisplayArea(ttk.Frame):
    """
    Фрейм для отображения текстовой информации.
    """
    def __init__(self, master, background_color="SystemButtonFace", *args, **kwargs):
        super().__init__(master, padding=(0,1,0,1), style="InfoArea.TFrame", *args, **kwargs)
        self.background_color = background_color

        self.columnconfigure(0, weight=1, uniform="infogroup")
        self.columnconfigure(1, weight=1, uniform="infogroup")
        self.columnconfigure(2, weight=1, uniform="infogroup")
        self.rowconfigure(0, weight=1)

        self._create_widgets()

    def _create_widgets(self):
        lf_padding = (3,2,3,3)
        label_pady = (1,1)
        label_font = ('TkDefaultFont', 8)
        wraplength_val = 190

        self.results_display_lf = ttk.LabelFrame(self, text="Результат U", padding=lf_padding)
        self.results_display_lf.grid(row=0, column=0, sticky=tk.NSEW, padx=(0,1))
        self.results_display_lf.rowconfigure(0, weight=1)
        self.results_display_lf.columnconfigure(0, weight=1)
        self.label_uniformity_results = ttk.Label(
            self.results_display_lf, text="Обновите графики",
            anchor=tk.NW, justify=tk.LEFT, wraplength=wraplength_val, font=label_font,
            background=self.background_color
        )
        self.label_uniformity_results.grid(row=0, column=0, sticky=tk.NSEW, padx=2, pady=label_pady)

        self.auto_uniformity_lf = ttk.LabelFrame(self, text="Авторавномерность", padding=lf_padding)
        self.auto_uniformity_lf.grid(row=0, column=1, sticky=tk.NSEW, padx=1)
        self.auto_uniformity_lf.rowconfigure(0, weight=1)
        self.auto_uniformity_lf.columnconfigure(0, weight=1)
        self.label_auto_uniformity_info = ttk.Label(
            self.auto_uniformity_lf, text="Результаты/статус...",
            anchor=tk.NW, justify=tk.LEFT, wraplength=wraplength_val, font=label_font,
            background=self.background_color
        )
        self.label_auto_uniformity_info.grid(row=0, column=0, sticky=tk.NSEW, padx=2, pady=label_pady)

        self.inverse_problem_lf = ttk.LabelFrame(self, text="Обратная Задача", padding=lf_padding)
        self.inverse_problem_lf.grid(row=0, column=2, sticky=tk.NSEW, padx=(1,0))
        self.inverse_problem_lf.rowconfigure(0, weight=1)
        self.inverse_problem_lf.columnconfigure(0, weight=1)
        self.label_inverse_problem_info = ttk.Label(
            self.inverse_problem_lf, text="Результаты/статус...",
            anchor=tk.NW, justify=tk.LEFT, wraplength=wraplength_val, font=label_font,
            background=self.background_color
        )
        self.label_inverse_problem_info.grid(row=0, column=0, sticky=tk.NSEW, padx=2, pady=label_pady)

    def update_uniformity_results(self, text: str):
        self.label_uniformity_results.config(text=text)
    def update_auto_uniformity_info(self, text: str):
        self.label_auto_uniformity_info.config(text=text)
    def update_inverse_problem_info(self, text: str):
        self.label_inverse_problem_info.config(text=text)

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Тест DisplayAreas (Layout Fix v4)")
    root.geometry("800x550")

    s = ttk.Style()
    bg_color = s.lookup("TFrame", "background")
    s.configure("InfoArea.TFrame", background=bg_color)

    main_frame = ttk.Frame(root, padding=10)
    main_frame.pack(fill=tk.BOTH, expand=True)
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(0, weight=3)
    main_frame.rowconfigure(1, weight=1)

    plot_area = PlotDisplayArea(main_frame, plot_size_pixels=280)
    plot_area.grid(row=0, column=0, sticky=tk.NSEW, pady=(0,5))

    # Добавляем тестовые данные на графики
    ax_map = plot_area.get_map_figure().add_subplot(111)
    ax_map.plot([0,1,2],[0,1,0], label="Карта")
    ax_map.set_title("Карта покрытия")
    ax_map.legend()

    ax_profile = plot_area.get_profile_figure().add_subplot(111)
    ax_profile.plot([0,1,2],[1,0,1], label="Профиль")
    ax_profile.set_title("Профиль")
    ax_profile.legend()
    
    # Важно: после добавления элементов на фигуры, нужно вызвать draw_canvases
    # В реальном приложении это будет делаться из ResultsWindow -> plot_manager
    # Здесь для теста вызовем напрямую, чтобы увидеть первоначальное состояние.
    # plot_area.draw_canvases() # Этот вызов теперь включает tight_layout

    info_area = InfoDisplayArea(main_frame, background_color=bg_color)
    info_area.grid(row=1, column=0, sticky=tk.NSEW)

    info_area.update_uniformity_results("U1: 10.5 %\nU3: 5.2 %")
    info_area.update_auto_uniformity_info("Маска рассчитана.\nРавномерность: 3.1 %")
    info_area.update_inverse_problem_info("Загружено: profile1.csv\nГотово к реконструкции.")
    
    # Чтобы увидеть эффект от <Configure>, можно добавить кнопку для изменения размера окна
    # или просто поресайзить окно вручную после запуска.
    # Для теста можно вызвать _apply_tight_layout_map и _apply_tight_layout_profile через некоторое время
    # root.after(1000, plot_area._apply_tight_layout_map)
    # root.after(1000, plot_area._apply_tight_layout_profile)


    root.mainloop()
