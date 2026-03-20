CREATE OR REPLACE VIEW v_user_activity AS
SELECT
    c.user_id,
    MIN(dt.turn_timestamp) AS first_seen_at,
    MAX(dt.turn_timestamp) AS last_seen_at,
    COUNT(DISTINCT c.chat_id) AS total_chats,
    COUNT(*) AS total_turns,
    AVG(CHAR_LENGTH(dt.prompt))::NUMERIC(10,2) AS avg_prompt_length,
    AVG(CHAR_LENGTH(COALESCE(dt.response, '')))::NUMERIC(10,2) AS avg_response_length
FROM chats c
JOIN dialogue_turns dt
    ON c.chat_id = dt.chat_id
GROUP BY c.user_id;

CREATE OR REPLACE VIEW v_chat_turn_counts AS
SELECT
    chat_id,
    COUNT(*) AS total_turns
FROM dialogue_turns
GROUP BY chat_id;

CREATE OR REPLACE VIEW v_dialogues_with_chat AS
SELECT
    dt.chat_id,
    dt.interaction_count,
    c.user_id,
    c.semester_code,
    c.chat_title,
    c.chat_start_time,
    dt.assignment_id,
    dt.turn_timestamp,
    dt.prompt,
    dt.response,
    dt.llm_label,
    dt.llm_justification
FROM dialogue_turns dt
JOIN chats c
    ON dt.chat_id = c.chat_id;

CREATE OR REPLACE VIEW v_submission_overview AS
SELECT
    s.user_id,
    s.assignment_id,
    s.directory_name,
    s.folder_path,
    COUNT(sf.relative_path) AS file_count,
    BOOL_OR(sf.file_type = 'ipynb') AS has_notebook,
    BOOL_OR(sf.file_type = 'py') AS has_python,
    SUM(COALESCE(sf.lines_of_code, 0)) AS total_lines_of_code,
    CASE
        WHEN BOOL_OR(sf.file_type = 'ipynb') AND BOOL_OR(sf.file_type = 'py') THEN 'notebook_and_python'
        WHEN BOOL_OR(sf.file_type = 'ipynb') THEN 'notebook'
        WHEN BOOL_OR(sf.file_type = 'py') THEN 'python'
        WHEN COUNT(sf.relative_path) = 0 THEN 'empty'
        ELSE 'mixed'
    END AS submitted_artifact_type
FROM submissions s
LEFT JOIN submission_files sf
    ON s.user_id = sf.user_id
   AND s.assignment_id = sf.assignment_id
GROUP BY
    s.user_id,
    s.assignment_id,
    s.directory_name,
    s.folder_path;

CREATE OR REPLACE VIEW v_user_grade_aggregates AS
SELECT
    ugs.user_id,
    a.semester_code,
    SUM(CASE WHEN a.assessment_kind = 'assignment'
             THEN ugs.normalized_score * a.max_points ELSE 0 END)
    / NULLIF(SUM(CASE WHEN a.assessment_kind = 'assignment'
                      THEN a.max_points ELSE 0 END), 0) AS llm,
    SUM(CASE WHEN a.assessment_kind = 'exam'
             THEN ugs.normalized_score * a.max_points ELSE 0 END)
    / NULLIF(SUM(CASE WHEN a.assessment_kind = 'exam'
                      THEN a.max_points ELSE 0 END), 0) AS no_llm
FROM user_assessment_scores ugs
JOIN assessments a
    ON ugs.assessment_id = a.assessment_id
GROUP BY
    ugs.user_id,
    a.semester_code;

CREATE OR REPLACE VIEW v_user_grades_wide AS
SELECT
    ugp.user_id,
    ugp.semester_code,
    ugp.directory_name,
    MAX(CASE WHEN a.assessment_code = 'a1' THEN ugs.normalized_score END) AS a1,
    MAX(CASE WHEN a.assessment_code = 'a2' THEN ugs.normalized_score END) AS a2,
    MAX(CASE WHEN a.assessment_code = 'a3' THEN ugs.normalized_score END) AS a3,
    MAX(CASE WHEN a.assessment_code = 'a4' THEN ugs.normalized_score END) AS a4,
    MAX(CASE WHEN a.assessment_code = 'a5' THEN ugs.normalized_score END) AS a5,
    MAX(CASE WHEN a.assessment_code = 'a6' THEN ugs.normalized_score END) AS a6,
    MAX(CASE WHEN a.assessment_code = 'a7' THEN ugs.normalized_score END) AS a7,
    MAX(CASE WHEN a.assessment_code = 'e1' THEN ugs.normalized_score END) AS e1,
    MAX(CASE WHEN a.assessment_code = 'e2' THEN ugs.normalized_score END) AS e2,
    MAX(CASE WHEN a.assessment_code = 'e3' THEN ugs.normalized_score END) AS e3,
    uga.llm,
    uga.no_llm
FROM user_grade_profiles ugp
LEFT JOIN user_assessment_scores ugs
    ON ugp.user_id = ugs.user_id
LEFT JOIN assessments a
    ON ugs.assessment_id = a.assessment_id
   AND ugp.semester_code = a.semester_code
LEFT JOIN v_user_grade_aggregates uga
    ON ugp.user_id = uga.user_id
   AND ugp.semester_code = uga.semester_code
GROUP BY
    ugp.user_id,
    ugp.semester_code,
    ugp.directory_name,
    uga.llm,
    uga.no_llm;

CREATE OR REPLACE VIEW v_assignment_activity AS
SELECT
    a.assignment_id,
    a.semester_code,
    a.assignment_code,
    a.title,
    COUNT(DISTINCT dt.chat_id) AS chats_about_assignment,
    COUNT(DISTINCT (dt.chat_id, dt.interaction_count)) AS total_turns_about_assignment,
    COUNT(DISTINCT s.user_id) AS users_with_submission,
    COUNT(DISTINCT (s.user_id, s.assignment_id)) AS total_submissions
FROM assignments a
LEFT JOIN dialogue_turns dt
    ON a.assignment_id = dt.assignment_id
LEFT JOIN submissions s
    ON a.assignment_id = s.assignment_id
GROUP BY
    a.assignment_id,
    a.semester_code,
    a.assignment_code,
    a.title;