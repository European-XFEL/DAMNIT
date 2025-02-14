
import numpy as np
from damnit.gui.widgets import PlotLineWidget, ValueRangeWidget


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


def test_value_range_widget(qtbot):
    # Test normal case with range
    values = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    vmin, vmax = 1.0, 3.0
    widget = ValueRangeWidget(values, vmin, vmax)
    qtbot.addWidget(widget)

    # Test initialization
    assert widget.min == vmin
    assert widget.max == vmax
    assert widget.plot is not None
    assert widget.slider is not None

    # Test setting values
    new_min, new_max = 1.5, 2.5
    with qtbot.waitSignal(widget.rangeChanged) as blocker:
        widget.set_values(new_min, new_max)
    assert len(blocker.args) == 2
    assert blocker.args[0] == new_min
    assert blocker.args[1] == new_max
    assert widget.min == new_min
    assert widget.max == new_max

    # Test input field changes
    widget.min_input.setText("2.0")
    with qtbot.waitSignal(widget.rangeChanged) as blocker:
        widget.min_input.editingFinished.emit()
    assert len(blocker.args) == 2
    assert blocker.args[0] == 2.0
    assert blocker.args[1] == 2.5
    assert widget.min == 2.0
    assert widget.plot.slider_position == (2.0, 2.5)

    # Test slider changes
    with qtbot.waitSignal(widget.rangeChanged) as blocker:
        widget.slider.setValue((1.8, 2.8))
    assert len(blocker.args) == 2
    assert blocker.args[0] == 1.8
    assert blocker.args[1] == 2.8
    assert widget.min_input.text() == "1.8"
    assert widget.max_input.text() == "2.8"

    # Test edge case: single value
    single_widget = ValueRangeWidget(np.array([1.0]), 1.0, 1.0)
    qtbot.addWidget(single_widget)
    assert single_widget.plot is None
    assert single_widget.slider is None
    assert single_widget.min == 1.0
    assert single_widget.max == 1.0
