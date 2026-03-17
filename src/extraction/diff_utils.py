import difflib

differ = difflib.Differ()

def extract_diff(template_lines, student_lines):
    """
    Returns only lines that are present in the student file but not in the template.
    Lines are returned as a list of strings without the '+ ' prefix.
    """
    diff = differ.compare(template_lines, student_lines)
    student_added_lines = [line[2:] for line in diff if line.startswith("+ ")]
    return student_added_lines