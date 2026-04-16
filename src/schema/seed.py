from pydantic import BaseModel

class SeedExample(BaseModel):
    """
    A single seed example to inject into prompt.
    """
    instruction: str
    input: str
    output: str
    