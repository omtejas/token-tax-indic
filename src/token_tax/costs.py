from __future__ import annotations

def estimate_api_input_cost(total_tokens: int, price_per_million_input_tokens: float, model_id: str, pricing_source: str, checked_date: str) -> dict:
    """Declared-rate simulation only; never infer a provider price."""
    if total_tokens < 0 or price_per_million_input_tokens < 0:
        raise ValueError("Token count and price must be non-negative")
    if not all([model_id.strip(), pricing_source.strip(), checked_date.strip()]):
        raise ValueError("Model, pricing source URL and checked date are required")
    return {"status":"simulated_not_billed","model_id":model_id,"input_tokens":int(total_tokens),"price_per_million_input_tokens":float(price_per_million_input_tokens),"estimated_input_cost":total_tokens*price_per_million_input_tokens/1_000_000,"pricing_source":pricing_source,"checked_date":checked_date}
