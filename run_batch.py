import os
import json
import uuid
from pathlib import Path
import config
from agents.orchestrator import LangGraphOrchestrator
from utils.teacher_turn_export import build_teacher_turn_record, save_teacher_turn_json

BATCH_INPUT_DIR = Path("batch_input")
BATCH_OUTPUT_DIR = Path("batch_output")
MCQ_OUTPUT_DIR = BATCH_OUTPUT_DIR / "mcq"
CQ_OUTPUT_DIR = BATCH_OUTPUT_DIR / "srijonshil"


def ensure_dirs():
    """Ensure all required batch directories exist at the root level."""
    for d in [BATCH_INPUT_DIR, MCQ_OUTPUT_DIR, CQ_OUTPUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def process_batch():
    """Reads JSON files from batch_input and generates assessments."""
    ensure_dirs()
    
    input_files = list(BATCH_INPUT_DIR.glob("*.json"))
    if not input_files:
        print(f"No JSON files found in {BATCH_INPUT_DIR.absolute()}")
        return

    orchestrator = LangGraphOrchestrator()

    for input_file in input_files:
        print(f"Processing batch file: {input_file.name}")
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            items = data if isinstance(data, list) else [data]
            
            for i, item in enumerate(items):
                query = item.get("query")
                level = item.get("level", "intermediate")
                session = item.get("session", f"batch_{input_file.stem}")
                tasks = item.get("tasks", [])
                
                if not query:
                    print(f"Item {i} in {input_file.name} missing 'query'. Skipping.")
                    continue
                
                print(f"  -> Processing query: {query[:50]}...")
                
                run_id = f"batch_{uuid.uuid4().hex[:8]}"
                state = orchestrator.process_query(query, thread_id=run_id, selected_tasks=tasks or None)
                
                path, rec, log_path = save_teacher_turn_json(
                    state,
                    thread_id=run_id,
                    run_id=run_id,
                    session_id=session
                )
                
                assessment = state.get("assessment") or {}
                mcqs = assessment.get("mcqs", [])
                if mcqs:
                    mcq_file = MCQ_OUTPUT_DIR / f"{run_id}_mcq.json"
                    with open(mcq_file, "w", encoding="utf-8") as f_mcq:
                        json.dump({
                            "run_id": run_id, 
                            "query": query, 
                            "difficulty": level,
                            "mcqs": mcqs
                        }, f_mcq, ensure_ascii=False, indent=2)
                
                cqs = assessment.get("creative_questions", [])
                if cqs:
                    cq_file = CQ_OUTPUT_DIR / f"{run_id}_srijonshil.json"
                    with open(cq_file, "w", encoding="utf-8") as f_cq:
                        json.dump({
                            "run_id": run_id, 
                            "query": query, 
                            "difficulty": level,
                            "creative_questions": cqs
                        }, f_cq, ensure_ascii=False, indent=2)
                
                print(f"     Done. Results saved to {BATCH_OUTPUT_DIR.name}/")
                
        except Exception as e:
            print(f"Error processing {input_file.name}: {str(e)}")

if __name__ == "__main__":
    process_batch()
