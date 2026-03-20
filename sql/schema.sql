CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE semesters (
    semester_code TEXT PRIMARY KEY,
    semester_name TEXT NOT NULL
);

INSERT INTO semesters (semester_code, semester_name)
VALUES
    ('f24', 'Fall 2024'),
    ('s25', 'Spring 2025')
ON CONFLICT (semester_code) DO NOTHING;

CREATE TABLE users (
    user_id TEXT PRIMARY KEY
);

CREATE TABLE assignments (
    assignment_id TEXT PRIMARY KEY,
    semester_code TEXT NOT NULL REFERENCES semesters(semester_code),
    assignment_code TEXT NOT NULL,
    title TEXT,
    description TEXT,
    folder_path TEXT,
    CONSTRAINT uq_assignments_semester_code UNIQUE (semester_code, assignment_code)
);

CREATE TABLE chats (
    chat_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    semester_code TEXT NOT NULL REFERENCES semesters(semester_code),
    chat_title TEXT,
    chat_start_time TIMESTAMP NULL
);

CREATE TABLE dialogue_turns (
    chat_id TEXT NOT NULL,
    interaction_count INTEGER NOT NULL,
    assignment_id TEXT NULL REFERENCES assignments(assignment_id) ON DELETE SET NULL,
    turn_timestamp TIMESTAMP NULL,
    prompt TEXT NOT NULL,
    response TEXT,
    llm_label TEXT,
    llm_justification TEXT,
    PRIMARY KEY (chat_id, interaction_count),
    CONSTRAINT fk_dialogue_turns_chat
        FOREIGN KEY (chat_id) REFERENCES chats(chat_id)
        ON DELETE CASCADE,
    CONSTRAINT chk_interaction_count
        CHECK (interaction_count >= 0)
);

CREATE TABLE dialogue_messages (
    chat_id TEXT NOT NULL,
    interaction_count INTEGER NOT NULL,
    message_sequence INTEGER NOT NULL,
    message_role TEXT,
    message_content TEXT NOT NULL,
    PRIMARY KEY (chat_id, interaction_count, message_sequence),
    CONSTRAINT fk_dialogue_messages_turn
        FOREIGN KEY (chat_id, interaction_count)
        REFERENCES dialogue_turns(chat_id, interaction_count)
        ON DELETE CASCADE,
    CONSTRAINT chk_message_sequence
        CHECK (message_sequence >= 1)
);

CREATE TABLE assignment_files (
    assignment_id TEXT NOT NULL REFERENCES assignments(assignment_id) ON DELETE CASCADE,
    relative_path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_type TEXT,
    extracted_text TEXT,
    PRIMARY KEY (assignment_id, relative_path)
);

CREATE TABLE submissions (
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    assignment_id TEXT NOT NULL REFERENCES assignments(assignment_id) ON DELETE CASCADE,
    directory_name TEXT,
    folder_path TEXT,
    PRIMARY KEY (user_id, assignment_id)
);

CREATE TABLE submission_files (
    user_id TEXT NOT NULL,
    assignment_id TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_type TEXT,
    extracted_text TEXT,
    file_size_bytes BIGINT,
    lines_of_code INTEGER,
    PRIMARY KEY (user_id, assignment_id, relative_path),
    CONSTRAINT fk_submission_files_submission
        FOREIGN KEY (user_id, assignment_id)
        REFERENCES submissions(user_id, assignment_id)
        ON DELETE CASCADE,
    CONSTRAINT chk_file_size
        CHECK (file_size_bytes >= 0 OR file_size_bytes IS NULL),
    CONSTRAINT chk_lines_of_code
        CHECK (lines_of_code >= 0 OR lines_of_code IS NULL)
);

CREATE TABLE assessments (
    assessment_id TEXT PRIMARY KEY,
    semester_code TEXT NOT NULL REFERENCES semesters(semester_code),
    assessment_code TEXT NOT NULL,
    assessment_kind TEXT NOT NULL,
    max_points NUMERIC(10,2) NOT NULL,
    assignment_id TEXT NULL REFERENCES assignments(assignment_id) ON DELETE SET NULL,
    CONSTRAINT uq_assessments_semester_code UNIQUE (semester_code, assessment_code),
    CONSTRAINT chk_assessment_kind CHECK (assessment_kind IN ('assignment', 'exam')),
    CONSTRAINT chk_max_points CHECK (max_points > 0)
);

CREATE TABLE user_grade_profiles (
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    semester_code TEXT NOT NULL REFERENCES semesters(semester_code),
    directory_name TEXT,
    PRIMARY KEY (user_id, semester_code)
);

CREATE TABLE user_assessment_scores (
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    assessment_id TEXT NOT NULL REFERENCES assessments(assessment_id) ON DELETE CASCADE,
    normalized_score NUMERIC(8,4),
    PRIMARY KEY (user_id, assessment_id)
);

CREATE INDEX idx_assignments_semester_code
    ON assignments(semester_code);

CREATE INDEX idx_chats_user_id
    ON chats(user_id);

CREATE INDEX idx_chats_semester_code
    ON chats(semester_code);

CREATE INDEX idx_dialogue_turns_assignment_id
    ON dialogue_turns(assignment_id);

CREATE INDEX idx_dialogue_turns_timestamp
    ON dialogue_turns(turn_timestamp);

CREATE INDEX idx_dialogue_turns_prompt_trgm
    ON dialogue_turns USING GIN (prompt gin_trgm_ops);

CREATE INDEX idx_dialogue_turns_response_trgm
    ON dialogue_turns USING GIN (response gin_trgm_ops);

CREATE INDEX idx_dialogue_messages_role
    ON dialogue_messages(message_role);

CREATE INDEX idx_assignment_files_file_type
    ON assignment_files(file_type);

CREATE INDEX idx_submission_files_file_type
    ON submission_files(file_type);

CREATE INDEX idx_assessments_semester_kind
    ON assessments(semester_code, assessment_kind);

CREATE INDEX idx_user_assessment_scores_assessment_id
    ON user_assessment_scores(assessment_id);