import argparse
from pathlib import Path
from huggingface_hub import HfApi

parser = argparse.ArgumentParser()
parser.add_argument("--repo", required=True, help="usuario/nome-do-modelo")
parser.add_argument("--space", help="usuario/nome-da-demo")
args = parser.parse_args()
root = Path(__file__).resolve().parent
api = HfApi()
api.create_repo(repo_id=args.repo, repo_type="model", exist_ok=True)
api.upload_folder(repo_id=args.repo, repo_type="model", folder_path=str(root / "model"))
if args.space:
    api.create_repo(repo_id=args.space, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(repo_id=args.space, repo_type="space", folder_path=str(root),
                      allow_patterns=["app.py", "predict.py", "requirements.txt", "model/*", "example.json"])
    api.upload_file(repo_id=args.space, repo_type="space", path_or_fileobj=str(root / "SPACE.md"),
                    path_in_repo="README.md")
print("https://huggingface.co/" + args.repo)
