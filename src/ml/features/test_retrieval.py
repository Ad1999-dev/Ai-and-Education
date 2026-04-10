import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.db import load_engine
from ml.features.dialogue import build_assignment_embedding_pairs
from ml.features.submission import build_submission_text_features
from ml.features.embedder import compute_similarity_features



def test_mvp_retrieval():
    # 1. Initialize the engine
    engine = load_engine()
    
    print("\n" + "="*60)
    print("--- Executing MVP Retrieval & Similarity for User 011bb520 ---")
    print("="*60)

    # 2. Fetch Data
    df_dialogues = build_assignment_embedding_pairs(engine)
    df_submissions = build_submission_text_features(engine)

    print(f"\n[Dialogue] Found {len(df_dialogues)} assignment interaction sets.")
    print(f"[Submission] Found {len(df_submissions)} assignment file sets.")

    # 3. Verify Alignment
    common_indices = df_dialogues.index.intersection(df_submissions.index)
    
    if len(common_indices) == 0:
        print("\n!!! WARNING: No matching records found between Dialogue and Submissions.")
        print("Check if the assignment_id strings match exactly in both tables.")
        return

    print(f"\n[Match] Found {len(common_indices)} assignments ready for GPU embedding.")

    # 4. Run the Transformer Embedding on GPU
    print("\n--- Initializing BGE-M3 and Computing Max Similarity ---")
    try:
        results_df = compute_similarity_features(df_dialogues, df_submissions)
        
        # 5. Final Display
        print("\n" + "X"*50)
        print("FINAL PROCESSED FEATURES")
        print("X"*50)
        print(results_df.to_string(index=False))
        print("X"*50)
        
        # Save results locally 
        results_df.to_csv("user_011bb520_features.csv", index=False)
        print(f"\nSaved {len(results_df)} features to 'user_011bb520_features.csv'")

    except Exception as e:
        print(f"\nERROR during embedding process: {e}")

if __name__ == "__main__":
    test_mvp_retrieval()