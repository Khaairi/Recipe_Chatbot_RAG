from pathlib import Path
# project_root = current_script_path.parent
db_dir = Path(__file__).resolve().parent / "chroma_db_data"

db_dir.mkdir(exist_ok=True)

print(db_dir)