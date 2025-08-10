__version__ = "0.1.0"
from .main import (
    main,
    create_app,
    classify_text,
    extract_text_from_eml,
    clean_email_text,
    html_to_text,
)
__all__ = ["main","create_app","classify_text","extract_text_from_eml","clean_email_text","html_to_text"]
