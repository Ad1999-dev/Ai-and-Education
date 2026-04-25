import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import torch
from sqlalchemy import text 

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from ml.db import load_engine
from ml.features.dialogue import build_assignment_embedding_pairs
from ml.features.submission import build_submission_text_features
from ml.features.embedder import compute_similarity_features

def get_all_student_ids(engine):
    """Fetches all unique student IDs from the StudyChat dataset."""
    id_query = text("SELECT DISTINCT user_id FROM public.chats")
    with engine.connect() as conn:
        result = conn.execute(id_query)
        return [row[0] for row in result]

def run_full_similarity_processing():
    engine = load_engine()
    
    # 1. FETCH ALL STUDENT IDs
    logging.info("Fetching student list...")
    all_ids = get_all_student_ids(engine) 
    
    # 2. FETCH ALL DATA AT ONCE
    logging.info(f"Fetching StudyChat data for {len(all_ids)} students...")
    df_dialogues = build_assignment_embedding_pairs(engine, user_ids=all_ids) 
    df_submissions = build_submission_text_features(engine, user_ids=all_ids)

    if df_dialogues.empty or df_submissions.empty:
        logging.error("No data found. Check your database connections.")
        return

    logging.info(f"Loaded {len(df_dialogues)} dialogue sets and {len(df_submissions)} submissions.")

    try:
        # 3. RUN OPTIMIZED EMBEDDER
        logging.info("Starting Bulk Similarity Analysis (GPU-accelerated)...")
        final_results_df, final_embeddings = compute_similarity_features(df_dialogues, df_submissions)

        # 4. SINGLE TRANSACTION SAVE
        if not final_results_df.empty:
            final_results_df.to_sql('similarity_features', engine, if_exists='replace', index=False)
            
            data_dir = PROJECT_ROOT / "data"
            data_dir.mkdir(exist_ok=True)
            
            final_results_df.to_csv(data_dir / "final_similarity_results.csv", index=False)
            np.savez_compressed(data_dir / "raw_embeddings.npz", embeddings=final_embeddings)

            print("\n" + "="*50)
            print("STUDYCHAT DATASET PROCESSED IN BULK")
            print(f"Total Similarity Pairs: {len(final_results_df)}")
            print("="*50)

    except Exception as e:
        logging.error(f"Processing failed: {e}")

if __name__ == "__main__":
    run_full_similarity_processing()