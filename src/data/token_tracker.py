from src.config import TOKEN_PRICE


class TokenTracker:
    def __init__(self, model: str):
        self.model = model
        self.input_tokens = 0
        self.output_tokens = 0
    
    def update(self, input_tokens: int, output_tokens: int):
        """
        Update token tracker with usage amounts from API call.
        """
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
    
    def get_total_cost(self) -> tuple[float, float]:
        """
        Return tuple of total input and output token costs.
        """
        total_input_token_cost = self.input_tokens / 1_000_000 * TOKEN_PRICE[self.model]["input"]
        total_output_token_cost = self.output_tokens / 1_000_000 * TOKEN_PRICE[self.model]["output"]
        return (total_input_token_cost, total_output_token_cost)