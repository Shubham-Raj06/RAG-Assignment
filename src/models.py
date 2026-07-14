from pydantic import BaseModel, Field
from typing import List, Optional

class CounselorResponse(BaseModel):
    answer: str = Field(description="The natural language answer to the user's question, strictly grounded in the context.")
    citations: List[str] = Field(description="List of college_ids cited in the answer.")
    answered: bool = Field(description="Whether the query was successfully answered or refused due to lack of matching information.")
    reason_if_unanswered: Optional[str] = Field(default=None, description="Explanation if the query could not be answered.")
