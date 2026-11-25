"""
IBM Cloud Object Storage service — cache raw data and reports.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)

_cos_client: Any = None


def _get_cos_client() -> Any:
    global _cos_client
    if _cos_client is not None:
        return _cos_client

    if settings.demo_mode or not settings.cos_api_key:
        logger.info("COS disabled (demo mode or no key) — using no-op storage")
        return None

    try:
        import ibm_boto3  # type: ignore
        from ibm_botocore.client import Config  # type: ignore

        _cos_client = ibm_boto3.client(
            "s3",
            ibm_api_key_id=settings.cos_api_key,
            ibm_service_instance_id=settings.cos_instance_crn,
            config=Config(signature_version="oauth"),
            endpoint_url=settings.cos_endpoint,
        )
        logger.info("IBM COS client initialised")
        return _cos_client
    except Exception as exc:
        logger.error("COS client init failed: %s", exc)
        return None


def _ensure_bucket(client: Any) -> None:
    try:
        client.head_bucket(Bucket=settings.cos_bucket)
    except Exception:
        try:
            client.create_bucket(Bucket=settings.cos_bucket)
            logger.info("Created COS bucket: %s", settings.cos_bucket)
        except Exception as exc:
            logger.warning("Could not create bucket: %s", exc)


def save_to_cos(key: str, data: Any) -> bool:
    """Serialise `data` as JSON and upload to COS. Returns True on success."""
    client = _get_cos_client()
    if client is None:
        logger.debug("COS unavailable — skipping save for key '%s'", key)
        return False
    try:
        _ensure_bucket(client)
        body = json.dumps(data, default=str).encode("utf-8")
        client.put_object(Bucket=settings.cos_bucket, Key=key, Body=body)
        logger.info("COS save OK: %s", key)
        return True
    except Exception as exc:
        logger.warning("COS save failed for '%s': %s", key, exc)
        return False


def load_from_cos(key: str) -> Any | None:
    """Load and deserialise JSON from COS. Returns None on miss/error."""
    client = _get_cos_client()
    if client is None:
        return None
    try:
        resp = client.get_object(Bucket=settings.cos_bucket, Key=key)
        body = resp["Body"].read()
        return json.loads(body)
    except client.exceptions.NoSuchKey:  # type: ignore[attr-defined]
        return None
    except Exception as exc:
        logger.warning("COS load failed for '%s': %s", key, exc)
        return None


def delete_from_cos(key: str) -> bool:
    client = _get_cos_client()
    if client is None:
        return False
    try:
        client.delete_object(Bucket=settings.cos_bucket, Key=key)
        return True
    except Exception as exc:
        logger.warning("COS delete failed for '%s': %s", key, exc)
        return False


def list_cos_keys(prefix: str = "") -> list[str]:
    client = _get_cos_client()
    if client is None:
        return []
    try:
        resp = client.list_objects_v2(Bucket=settings.cos_bucket, Prefix=prefix)
        return [obj["Key"] for obj in resp.get("Contents", [])]
    except Exception as exc:
        logger.warning("COS list failed: %s", exc)
        return []
