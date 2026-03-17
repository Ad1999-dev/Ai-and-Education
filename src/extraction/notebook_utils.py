import json
from .diff_utils import extract_diff

def process_notebook(template_file, student_file):
    """
    Extract student-added lines from a Jupyter notebook (.ipynb)
    - Only compares code cells
    - Ignores outputs, markdown, and metadata
    """

    # Load notebooks
    with open(template_file, "r", encoding="utf-8") as f:
        template_nb = json.load(f)
    with open(student_file, "r", encoding="utf-8") as f:
        student_nb = json.load(f)

    # Extract code cells
    template_cells = [cell["source"] for cell in template_nb["cells"] if cell["cell_type"] == "code"]
    student_cells = [cell["source"] for cell in student_nb["cells"] if cell["cell_type"] == "code"]

    # Flatten the list of lists into single list of lines
    template_lines = [line for cell in template_cells for line in cell]
    student_lines = [line for cell in student_cells for line in cell]

    # Use diff_utils to extract student-added lines
    return extract_diff(template_lines, student_lines)