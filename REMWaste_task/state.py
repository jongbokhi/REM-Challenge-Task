from typing import List
from typing_extensions import TypedDict, Annotated


# Define State
class GraphState(TypedDict):
    """
    Graph state data model

    Attributes:
        question: user input
        documents: List of document
        accent_type: Classification of the accent
        accent_grade: Numeric confidence score representing the strength of the English accent
        generation: LLM generated answer
    """

    question: Annotated[str, "User question"]
    documents: Annotated[List[str], "List of documents"]
    accent_type: Annotated[
        str, "Classification of the accent (e.g., British, American, Australian, etc.)"
    ]
    accent_grade: Annotated[float, "English accent confidence score"]
    generation: Annotated[str, "LLM generated answer"]
