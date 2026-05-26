from huggingface_hub import snapshot_download, login

repo_id = "SprintML/tml26_task2"

authToken = "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"

login(token = authToken)

snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    allow_patterns=[
        "target_model/",
        "suspect_models/",
    ],
    local_dir_use_symlinks=False,
    resume_download=True,
    
)
print("Code files downloaded.")