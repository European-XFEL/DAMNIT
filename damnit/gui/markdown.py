import re

import pandas as pd


def dataframe_to_markdown(table: pd.DataFrame) -> str:
    """Format a dataframe as a compact Markdown table."""
    markdown = table.to_markdown(index=False, disable_numparse=True)
    lines = markdown.strip().splitlines()
    output_lines = []

    for line in lines:
        cells = line.split("|")
        processed_cells = [re.sub(r"\s+", " ", cell.strip()) for cell in cells]
        output_lines.append("|".join(processed_cells))

    return "\n".join(output_lines)
