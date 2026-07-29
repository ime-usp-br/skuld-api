# app/api/routers/__init__.py
"""Roteadores HTTP por versão da API."""

from app.api.routers import cache, v1, v2

__all__ = ["cache", "v1", "v2"]