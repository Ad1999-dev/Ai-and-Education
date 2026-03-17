import os
import shutil
from .diff_utils import extract_diff
from .notebook_utils import process_notebook

def process_py(template_file, student_file):
    """Extract only student-added lines from Python file."""
    with open(template_file, "r", encoding="utf-8") as f:
        template_lines = f.readlines()
    with open(student_file, "r", encoding="utf-8") as f:
        student_lines = f.readlines()
    return extract_diff(template_lines, student_lines)

def process_assignment(base_dir, template_dir, config, output_dir):
    """
    base_dir: path to assignment folder containing student folders
    template_dir: path to assignment template files
    config: dict with keys:
        - 'type': 'py' / 'notebook'
        - 'files': list of files OR list of dicts with 'template' and 'student' keys
    output_dir: path to save extracted student code
    """
    os.makedirs(output_dir, exist_ok=True)

    # Loop over all student folders
    for student in os.listdir(base_dir):
        student_folder = os.path.join(base_dir, student)
        if not os.path.isdir(student_folder):
            continue

        student_output_dir = os.path.join(output_dir, student)
        os.makedirs(student_output_dir, exist_ok=True)

        print(f"\nProcessing student: {student}")

        for file_info in config["files"]:
            # Handle dict (template vs student filename) or string (same name)
            if isinstance(file_info, dict):
                template_file_name = file_info["template"]
                student_file_name = file_info["student"]
            else:
                template_file_name = student_file_name = file_info

            student_file = os.path.join(student_folder, student_file_name)
            template_file = os.path.join(template_dir, template_file_name)

            if not os.path.exists(student_file):
                print(f"  Skipping {student_file_name}, not found for student {student} in {student_file}")
                continue
            if config["type"] != "md" and not os.path.exists(template_file):
                print(f"  Template {template_file_name} not found, skipping in {student_file_name} in template {template_dir}")
                continue

            # Process based on type
            if config["type"] == "py":
                result = process_py(template_file, student_file)
            elif config["type"] == "notebook":
                result = process_notebook(template_file, student_file)
            elif config["type"] == "md":
                shutil.copy(student_file, os.path.join(student_output_dir, student_file_name))
                print(f"  Copied {student_file_name}")
                continue
            else:
                print(f"  Unsupported file type {config['type']}")
                continue

            # Save extracted code
            output_file = os.path.join(student_output_dir, student_file_name)
            with open(output_file, "w", encoding="utf-8") as f:
                f.writelines(result)
            print(f"  Extracted {student_file_name}")