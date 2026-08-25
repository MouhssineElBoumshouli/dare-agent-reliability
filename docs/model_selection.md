# Phase-1 model selection

Status: pilot candidate; not yet frozen for final execution.

The pilot uses the exact OpenAI snapshot `gpt-4.1-mini-2025-04-14` through the
Chat Completions API. OpenAI describes GPT-4.1 mini as an instruction-following
and tool-calling model, documents function-calling support, and lists this dated
snapshot. The dated ID avoids a moving alias. The documented token prices at the
time of selection (2026-08-25) were $0.40 per million input tokens, $0.10 per
million cached input tokens, and $1.60 per million output tokens.

Primary source:
https://developers.openai.com/api/docs/models/gpt-4.1-mini

The choice is also constrained by the pinned DARE-Bench implementation. Its
generic OpenAI adapter uses Chat Completions, sends `temperature`, `top_p`, and
function definitions, and does not expose the newer reasoning-model request
surface. GPT-4.1 mini therefore provides a clean match without changing the
benchmark agent implementation. The pilot uses temperature 0.001 because the
pinned generation code replaces a numeric zero with 0.7 via a truthiness
fallback. This compatibility detail is recorded rather than silently patched.

Provider account access and quota remain unverified until a credential is
supplied. A successful pilot is required before this model can be frozen for the
240 final runs.
