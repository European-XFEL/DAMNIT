
import numpy as np
from damnit.gui.widgets import PlotLineWidget


def test_plot_line_widget(qtbot):
    # Create test data
    x_data = np.array([0, 1, 2, 3, 4])
    y_data = np.array([0, 1, 0, 1, 0])

    # Test initialization
    widget = PlotLineWidget(x_data, y_data)
    qtbot.addWidget(widget)

    assert np.array_equal(widget.x_data, x_data)
    assert np.array_equal(widget.y_data, y_data)
    assert widget.slider_position == (x_data[0], x_data[-1])

    # Test slider position updates
    new_position = (1.0, 3.0)
    widget.set_slider_position(new_position)
    assert widget.slider_position == new_position

    # Test painting
    widget.resize(200, 200)
    widget.show()
    widget.repaint()

    # Test edge cases
    widget.set_slider_position((x_data[0], x_data[-1]))
    widget.repaint()
