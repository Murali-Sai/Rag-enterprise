"""MCP (Model Context Protocol) server for the SEC EDGAR RAG system.

Exposes the RAG pipeline as MCP-native tools so any MCP client
(Claude Desktop, Cursor, etc.) can query real SEC 10-K filings,
with RBAC and financial compliance guardrails preserved.
"""
