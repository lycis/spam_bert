"""Spam BERT Detector package: re-export public API for tests/imports."""

__version__ = "0.1.0"

# Re-export everything the tests use from the implementation module
from .app import (  # type: ignore F401
    # config/constants
    DEFAULT_LOCAL_MODEL_DIR,
    # utilities
    html_to_text,
    clean_email_text,
    extract_text_from_eml,
    is_hub_id,
    sanitized_repo_dir,
    # model/inference
    ensure_local_copy,
    resolve_model_source,
    get_tokenizer,
    get_pipeline_cached,
    classify_text,
    # app wiring
    create_app,
    main,  # CLI entry
)

__all__ = [
    "DEFAULT_LOCAL_MODEL_DIR",
    "html_to_text",
    "clean_email_text",
    "extract_text_from_eml",
    "is_hub_id",
    "sanitized_repo_dir",
    "ensure_local_copy",
    "resolve_model_source",
    "get_tokenizer",
    "get_pipeline_cached",
    "classify_text",
    "create_app",
    "main",
]
