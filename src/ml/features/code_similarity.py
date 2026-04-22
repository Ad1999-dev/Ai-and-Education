import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy import text
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from ml.db import load_engine
from ml.features.dialogue import build_assignment_embedding_pairs
from ml.features.submission import build_submission_text_features
from ml.features.embedder import compute_similarity_features

def get_all_student_ids(engine):
    """Fetches all 203 unique student IDs who consented to the study[cite: 119]."""
    id_query = text("SELECT DISTINCT user_id FROM public.chats")
    with engine.connect() as conn:
        result = conn.execute(id_query)
        return [row[0] for row in result]

def chunk_list(data, size):
    """Yield successive n-sized chunks from data."""
    for i in range(0, len(data), size):
        yield data[i:i + size]

def run_full_similarity_processing(batch_size=25):
    engine = load_engine()
    all_student_ids = get_all_student_ids(engine)
    total_students = len(all_student_ids)
    
    logging.info(f"Processing StudyChat dataset: {total_students} students.")
    
    all_results = []
    all_vectors = [] # List to hold raw embedding arrays

    for i, batch_ids in enumerate(chunk_list(all_student_ids, batch_size)):
        logging.info(f"Processing Batch {i+1}...")
        
        try:
            # Retrieve data using DA labels: Conceptual, Writing, Context, etc. 
            df_dialogues = build_assignment_embedding_pairs(engine, user_ids=batch_ids)
            df_submissions = build_submission_text_features(engine, user_ids=batch_ids)

            if df_dialogues.empty or df_submissions.empty:
                continue

            # Modified: Returns a tuple of (DataFrame, Numpy_Array)
            batch_results, batch_embeddings = compute_similarity_features(df_dialogues, df_submissions)
            
            all_results.append(batch_results)
            all_vectors.append(batch_embeddings)
            
            # Save progress to DB incrementally to prevent data loss
            batch_results.to_sql('similarity_features', engine, if_exists='append', index=False)
            
        except Exception as e:
            logging.error(f"Error in batch {i+1}: {e}")

    # Final Consolidation
    if all_results:
        # 1. Save final CSV for analysis
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(PROJECT_ROOT / "data" / "final_similarity_results.csv", index=False)
        
        # 2. Save raw embeddings as compressed NumPy file
        final_embeddings = np.vstack(all_vectors)
        np.savez_compressed(PROJECT_ROOT / "data" / "raw_embeddings.npz", embeddings=final_embeddings)
        
        print("\n" + "="*50)
        print("STUDYCHAT DATASET PROCESSED")
        print(f"Total Similarity Pairs: {len(final_df)}")
        print(f"Embeddings Saved: {final_embeddings.shape}")
        print("="*50)

if __name__ == "__main__":
    run_full_similarity_processing()