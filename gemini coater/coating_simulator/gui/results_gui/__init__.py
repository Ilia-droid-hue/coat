# coating_simulator/gui/results_gui/__init__.py
"""
Инициализационный файл для пакета results_gui.
Этот пакет содержит модули, отвечающие за интерфейс окна результатов.
"""

# Импортируем основной класс окна результатов для удобства доступа
# например, from coating_simulator.gui.results_gui import ResultsWindow
from .results_window import ResultsWindow

# Можно также импортировать другие ключевые классы или функции, если это необходимо
# from .settings_panel import SettingsPanel
# from .display_areas import PlotDisplayArea, InfoDisplayArea
# from .profile_utils import extract_profiles_for_statistics
# from .plot_manager import update_plots
# from .action_callbacks import placeholder_export_excel

# Определяем __all__ для控制暴露되는 символов при from . import *
__all__ = ['ResultsWindow']
