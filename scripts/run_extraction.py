import os
from src.extraction.extractor import process_assignment


# Assignment configuration
ASSIGNMENTS = {
        "s25": {
        "a2": {
            "type": "py",
            "files": [
                {"template": "autocomplete.py", "student": "autocomplete.py"},
            ]
        },
        "a3": {
            "type": "notebook",
            "files": [
                {"template": "CS_383_Data_Cleaning_Assignment.ipynb", "student": "CS_383_Data_Cleaning_Assignment.ipynb"}
            ]
        },
        "a4": {
            "type": "notebook",
            "files": [
                {"template": "sklearn_sample_notebook.ipynb", "student": "sklearn_sample_notebook.ipynb"}
            ]
        },
        "a5": {
            "type": "notebook",
            "files": [
                {"template": "a5_S25_notebook.ipynb", "student": "PytorchStudentViewV1.ipynb"}
            ]
        },
        "a6": {
            "type": "py",
            "files": [
                {"template": "NgramAutocomplete.py", "student": "NgramAutocomplete.py"},

            ]
        },
        "a7": {
            "type": "py",
            "files": [
                {"template": "rnn_complete.py", "student": "rnn_complete.py"},
            ]
        }
    },
    "f24": {  
        "a2": {
            "type": "py",
            "files": [
                {"template": "autocomplete.py", "student": "autocomplete.py"},
            ]
        },
        "a3": {
            "type": "notebook",
            "files": [
                {"template": "a3_notebook.ipynb", "student": "CS_383_Data_Cleaning_Assignment.ipynb"}
            ]
        },
        "a4": {
            "type": "notebook",
            "files": [
                {"template": "slime_sample_notebook.ipynb", "student": "slime_sample_notebook.ipynb"}
            ]
        },
        "a5": {
            "type": "notebook",
            "files": [
                {"template": "flowers_sample_notebook.ipynb", "student": "flowers_sample_notebook.ipynb"}
            ]
        },
        "a6": {
            "type": "py",
            "files": [
                {"template": "NgramAutocomplete.py", "student": "NgramAutocomplete.py"},
            ]
        },
        "a7": {
            "type": "notebook",
            "files": [
                {"template": "a7_notebook.ipynb", "student": "ML.ipynb"}
            ]
        }
    },

}


# Map assignments to submission folder names
SUBMISSION_FOLDER_MAP = {  
    "s25": {
        "a2": "assignment-2-search-complete-submissions",
        "a3": "assignment-3-data-cleaning-and-preprocessing-submissions",
        "a4": "assignment-4-sklearn-for-machine-learning-submissions",
        "a5": "assignment-5-introduction-to-pytorch-submissions",
        "a6": "assignment-6-n-gram-complete-submissions",
        "a7": "assignment-7-neural-complete-submissions",
    },
    "f24": {
        "a2": "assignment-2-search-complete-submissions",
        "a3": "assignment-3-data-cleaning-and-preprocessing-submissions",
        "a4": "assignment-4-equation-of-a-slime-submissions",
        "a5": "assignment-5-judging-flowers-submissions",
        "a6": "assignment-6-n-gram-language-models-submissions",
        "a7": "assignment-7-rnns-ml-complete-submissions",
    },
  
}


# Paths
SEMESTERS = ["s25","f24"]
STUDENTS_ROOT = "data/raw/submissions/studychats_student_submissions_cleaned_final/studychats_student_submissions_cleaned_final"
TEMPLATES_ROOT = "data/raw/assignment_text"
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Extraction loop
for semester in SEMESTERS:
    print(f"\n=== Processing semester: {semester} ===")
    
    semester_students_dir = os.path.join(STUDENTS_ROOT, semester)
    semester_templates_dir = os.path.join(TEMPLATES_ROOT, f"studychat_{semester}_assignments")
    
    # Use semester-specific assignment mapping
    assignments_config = ASSIGNMENTS[semester]
    submission_folder_map = SUBMISSION_FOLDER_MAP[semester]

    for assignment_name, config in assignments_config.items():
        submission_folder_name = submission_folder_map.get(assignment_name)
        if not submission_folder_name:
            print(f"No submission folder mapping for {assignment_name} in {semester}, skipping")
            continue

        assignment_submission_dir = os.path.join(semester_students_dir, submission_folder_name)
        if not os.path.exists(assignment_submission_dir):
            print(f"Assignment folder not found: {assignment_submission_dir}")
            continue

        template_dir = os.path.join(semester_templates_dir, assignment_name)
        assignment_output_dir = os.path.join(OUTPUT_DIR, semester, assignment_name)
        os.makedirs(assignment_output_dir, exist_ok=True)

        print(f"\n--- Extracting {assignment_name} from {semester} ---")

        process_assignment(
            base_dir=assignment_submission_dir,
            template_dir=template_dir,
            config=config,
            output_dir=assignment_output_dir
        )