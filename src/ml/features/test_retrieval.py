import sys
from pathlib import Path
import pandas as pd
from sqlalchemy import text

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from ml.db import load_engine
from ml.features.dialogue import build_assignment_embedding_pairs
from ml.features.submission import build_submission_text_features
from ml.features.embedder import compute_similarity_features

def run_batch_similarity():
    engine = load_engine()
    
    # 1. Fetch the first 10 unique student IDs
    # We pull from chats to ensure we pick students who actually used the AI
    id_query = text("SELECT DISTINCT user_id FROM public.chats LIMIT 10")
    
    with engine.connect() as conn:
        result = conn.execute(id_query)
        student_ids = [row[0] for row in result]

    print(f"--- Fetched 10 Students for Processing ---")
    print(f"IDs: {[sid[:8] for sid in student_ids]}") 

    # 2. Retrieve all Dialogues for these 10 students
    df_dialogues = build_assignment_embedding_pairs(engine, user_ids=student_ids)
    
    # 3. Retrieve all Submissions for these 10 students
    df_submissions = build_submission_text_features(engine, user_ids=student_ids)

    print(f"\n[Data] Dialogues: {len(df_dialogues)} rows | Submissions: {len(df_submissions)} rows")

    # 4. Run the Transformer Embedding and Save Results
    if not df_dialogues.empty and not df_submissions.empty:
        results_df = compute_similarity_features(df_dialogues, df_submissions)
        
        print("\n" + "="*50)
        print("BATCH PROCESSING COMPLETE")
        print(f"Total Assignment Pairs Processed: {len(results_df)}")
        print("="*50)
        print(results_df)
    else:
        print("No matching data found for this batch.")

if __name__ == "__main__":
    run_batch_similarity()