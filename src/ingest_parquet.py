import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, List, Dict, Optional, Tuple, Union

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_PARQUET_DIR = PROJECT_ROOT / "data" / "raw" / "parquet"


def load_engine():
    load_dotenv(ENV_PATH)
    import os

    db_user = os.getenv("DB_USER", "studychat_user")
    db_password = os.getenv("DB_PASSWORD", "studychat_password")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "studychat")

    url = f"postgresql+psycopg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    return create_engine(url, future=True, pool_pre_ping=True)


def chunked(items: List[Dict], size: int) -> Iterable[List[Dict]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def safe_null(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def to_python_datetime(value: Any):
    value = safe_null(value)
    if value is None:
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()


def detect_messages_column(df: pd.DataFrame) -> Optional[str]:
    for col in ["messages", "message", "messages_json"]:
        if col in df.columns:
            return col
    return None


def build_assignment_id(semester: Any, topic: Any) -> Optional[str]:
    semester = safe_null(semester)
    topic = safe_null(topic)
    if semester is None or topic is None:
        return None

    semester_str = str(semester).strip().lower()
    topic_str = str(topic).strip().lower()

    if semester_str not in {"f24", "s25"}:
        return None
    if not re.fullmatch(r"a\d+", topic_str):
        return None

    return f"{semester_str}_{topic_str}"


def extract_llm_fields(row: pd.Series) -> Tuple[Optional[str], Optional[str]]:
    if "llm_label" in row.index:
        raw = safe_null(row["llm_label"])
        if isinstance(raw, dict):
            return safe_null(raw.get("label")), safe_null(raw.get("justification"))
        if isinstance(raw, str):
            raw_str = raw.strip()
            if raw_str.startswith("{") and raw_str.endswith("}"):
                try:
                    parsed = json.loads(raw_str)
                    return safe_null(parsed.get("label")), safe_null(parsed.get("justification"))
                except Exception:
                    return raw_str, None
            return raw_str, None

    label = None
    justification = None

    for col in ["label", "llm_label_label", "llm_label.label"]:
        if col in row.index:
            label = safe_null(row[col])
            break

    for col in ["justification", "llm_justification", "llm_label_justification", "llm_label.justification"]:
        if col in row.index:
            justification = safe_null(row[col])
            break

    return label, justification


def ensure_message_list(value: Any) -> List[Dict]:
    value = safe_null(value)
    if value is None:
        return []

    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, dict):
                result.append(item)
            else:
                result.append({"role": None, "content": str(item)})
        return result

    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith("[") and raw.endswith("]"):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    result = []
                    for item in parsed:
                        if isinstance(item, dict):
                            result.append(item)
                        else:
                            result.append({"role": None, "content": str(item)})
                    return result
            except Exception:
                pass

    return []


def parse_arguments():
    parser = argparse.ArgumentParser(description="Ingest StudyChat parquet files into normalized PostgreSQL schema.")
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    parser.add_argument("--batch-size", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_arguments()
    parquet_dir = args.parquet_dir
    batch_size = args.batch_size

    if not parquet_dir.exists():
        raise FileNotFoundError(f"Parquet directory not found: {parquet_dir}")

    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in: {parquet_dir}")

    engine = load_engine()

    semester_sql = text(
        """
        INSERT INTO semesters (semester_code, semester_name)
        VALUES (:semester_code, :semester_name)
        ON CONFLICT (semester_code) DO NOTHING
        """
    )

    users_sql = text(
        """
        INSERT INTO users (user_id)
        VALUES (:user_id)
        ON CONFLICT (user_id) DO NOTHING
        """
    )

    assignments_sql = text(
        """
        INSERT INTO assignments (assignment_id, semester_code, assignment_code, title, description, folder_path)
        VALUES (:assignment_id, :semester_code, :assignment_code, NULL, NULL, NULL)
        ON CONFLICT (assignment_id) DO NOTHING
        """
    )

    chats_sql = text(
        """
        INSERT INTO chats (chat_id, user_id, semester_code, chat_title, chat_start_time)
        VALUES (:chat_id, :user_id, :semester_code, :chat_title, :chat_start_time)
        ON CONFLICT (chat_id) DO UPDATE
        SET
            user_id = EXCLUDED.user_id,
            semester_code = EXCLUDED.semester_code,
            chat_title = COALESCE(EXCLUDED.chat_title, chats.chat_title),
            chat_start_time = COALESCE(EXCLUDED.chat_start_time, chats.chat_start_time)
        """
    )

    turns_sql = text(
        """
        INSERT INTO dialogue_turns (
            chat_id,
            interaction_count,
            assignment_id,
            turn_timestamp,
            prompt,
            response,
            llm_label,
            llm_justification
        )
        VALUES (
            :chat_id,
            :interaction_count,
            :assignment_id,
            :turn_timestamp,
            :prompt,
            :response,
            :llm_label,
            :llm_justification
        )
        ON CONFLICT (chat_id, interaction_count) DO UPDATE
        SET
            assignment_id = EXCLUDED.assignment_id,
            turn_timestamp = EXCLUDED.turn_timestamp,
            prompt = EXCLUDED.prompt,
            response = EXCLUDED.response,
            llm_label = EXCLUDED.llm_label,
            llm_justification = EXCLUDED.llm_justification
        """
    )

    messages_sql = text(
        """
        INSERT INTO dialogue_messages (
            chat_id,
            interaction_count,
            message_sequence,
            message_role,
            message_content
        )
        VALUES (
            :chat_id,
            :interaction_count,
            :message_sequence,
            :message_role,
            :message_content
        )
        ON CONFLICT (chat_id, interaction_count, message_sequence) DO UPDATE
        SET
            message_role = EXCLUDED.message_role,
            message_content = EXCLUDED.message_content
        """
    )

    semester_records = [
        {"semester_code": "f24", "semester_name": "Fall 2024"},
        {"semester_code": "s25", "semester_name": "Spring 2025"},
    ]

    total_turns = 0
    total_messages = 0

    with engine.begin() as conn:
        conn.execute(semester_sql, semester_records)

    for parquet_file in parquet_files:
        print(f"Reading {parquet_file.name} ...")
        df = pd.read_parquet(parquet_file)

        required = {"chatId", "userId", "semester", "interactionCount", "prompt"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{parquet_file.name} is missing required columns: {sorted(missing)}")

        messages_col = detect_messages_column(df)

        user_records = []
        for user_id in df["userId"].dropna().astype(str).drop_duplicates():
            user_records.append({"user_id": user_id})

        assignment_records = []
        seen_assignment_ids = set()
        if "topic" in df.columns:
            for _, row in df[["semester", "topic"]].drop_duplicates().iterrows():
                assignment_id = build_assignment_id(row["semester"], row["topic"])
                if assignment_id and assignment_id not in seen_assignment_ids:
                    seen_assignment_ids.add(assignment_id)
                    assignment_records.append(
                        {
                            "assignment_id": assignment_id,
                            "semester_code": assignment_id.split("_")[0],
                            "assignment_code": assignment_id.split("_")[1],
                        }
                    )

        chat_cols = [c for c in ["chatId", "userId", "semester", "chatTitle", "chatStartTime"] if c in df.columns]
        chats_df = df[chat_cols].drop_duplicates(subset=["chatId"])

        chat_records = []
        for _, row in chats_df.iterrows():
            chat_id = safe_null(row.get("chatId"))
            user_id = safe_null(row.get("userId"))
            semester_code = safe_null(row.get("semester"))
            if chat_id is None or user_id is None or semester_code is None:
                continue

            chat_records.append(
                {
                    "chat_id": str(chat_id),
                    "user_id": str(user_id),
                    "semester_code": str(semester_code).lower(),
                    "chat_title": safe_null(row.get("chatTitle")),
                    "chat_start_time": to_python_datetime(row.get("chatStartTime")),
                }
            )

        turn_records = []
        message_records = []

        for _, row in df.iterrows():
            chat_id = safe_null(row.get("chatId"))
            interaction_count = safe_null(row.get("interactionCount"))
            prompt = safe_null(row.get("prompt"))

            if chat_id is None or interaction_count is None or prompt is None:
                continue

            llm_label, llm_justification = extract_llm_fields(row)
            assignment_id = build_assignment_id(row.get("semester"), row.get("topic"))

            turn_records.append(
                {
                    "chat_id": str(chat_id),
                    "interaction_count": int(interaction_count),
                    "assignment_id": assignment_id,
                    "turn_timestamp": to_python_datetime(row.get("timestamp")),
                    "prompt": str(prompt),
                    "response": safe_null(row.get("response")),
                    "llm_label": llm_label,
                    "llm_justification": llm_justification,
                }
            )

            if messages_col is not None:
                messages = ensure_message_list(row.get(messages_col))
                for i, message in enumerate(messages, start=1):
                    message_records.append(
                        {
                            "chat_id": str(chat_id),
                            "interaction_count": int(interaction_count),
                            "message_sequence": i,
                            "message_role": safe_null(message.get("role")),
                            "message_content": str(message.get("content", "")),
                        }
                    )

        with engine.begin() as conn:
            for batch in chunked(user_records, batch_size):
                conn.execute(users_sql, batch)

            for batch in chunked(assignment_records, batch_size):
                conn.execute(assignments_sql, batch)

            for batch in chunked(chat_records, batch_size):
                conn.execute(chats_sql, batch)

            for batch in chunked(turn_records, batch_size):
                conn.execute(turns_sql, batch)

            for batch in chunked(message_records, batch_size):
                conn.execute(messages_sql, batch)

        total_turns += len(turn_records)
        total_messages += len(message_records)
        print(
            f"Inserted/updated {len(user_records)} users, "
            f"{len(chat_records)} chats, {len(turn_records)} turns, "
            f"{len(message_records)} messages from {parquet_file.name}."
        )

    print(f"Done. Total turns processed: {total_turns}. Total messages processed: {total_messages}.")


if __name__ == "__main__":
    main()