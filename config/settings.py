"""
InvestIQ – centralised configuration loaded from environment / .env file.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Demo mode ────────────────────────────────────────────────────────────
    demo_mode: bool = Field(default=False, alias="DEMO_MODE")

    # ── watsonx.ai / IBM Granite ──────────────────────────────────────────────
    watsonx_api_key: str = Field(default="", alias="WATSONX_API_KEY")
    watsonx_project_id: str = Field(default="", alias="WATSONX_PROJECT_ID")
    watsonx_url: str = Field(
        default="https://us-south.ml.cloud.ibm.com",
        alias="WATSONX_URL",
    )
    granite_model_id: str = Field(
        default="ibm/granite-3-8b-instruct",
        alias="GRANITE_MODEL_ID",
    )

    # ── IBM Cloud Object Storage ──────────────────────────────────────────────
    cos_api_key: str = Field(default="", alias="COS_API_KEY")
    cos_instance_crn: str = Field(default="", alias="COS_INSTANCE_CRN")
    cos_endpoint: str = Field(
        default="https://s3.us-south.cloud-object-storage.appdomain.cloud",
        alias="COS_ENDPOINT",
    )
    cos_bucket: str = Field(default="investiq-cache", alias="COS_BUCKET")

    # ── News API ──────────────────────────────────────────────────────────────
    news_api_key: str = Field(default="", alias="NEWS_API_KEY")

    # ── Application ──────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    retry_delay: float = Field(default=1.0, alias="RETRY_DELAY")
    pipeline_timeout: int = Field(default=300, alias="PIPELINE_TIMEOUT")
    granite_call_timeout: int = Field(default=60, alias="GRANITE_CALL_TIMEOUT")
    granite_json_retries: int = Field(default=2, alias="GRANITE_JSON_RETRIES")

    # ── RAG / Vector store ───────────────────────────────────────────────────
    rag_chunk_size: int = Field(default=800, alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=100, alias="RAG_CHUNK_OVERLAP")
    rag_top_k: int = Field(default=5, alias="RAG_TOP_K")
    rag_index_dir: str = Field(default=".rag_index", alias="RAG_INDEX_DIR")

    # ── FastAPI ───────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")


settings = Settings()
