from state import GraphState
from abc import ABC, abstractmethod
from chains import (
    create_language_checker_chain,
    create_accent_grader_chain,
    create_accent_classifier_chain,
)


# ===============BaseNode#===============#
class BaseNode(ABC):
    def __init__(self, **kwargs):
        self.name = "BaseNode"
        self.verbose = False
        if "verbose" in kwargs:
            self.verbose = kwargs["verbose"]

    @abstractmethod
    def execute(self, state: GraphState) -> GraphState:
        pass

    def logging(self, method_name, **kwargs):
        if self.verbose:
            print(f"[{self.name}] {method_name}")
            for key, value in kwargs.items():
                print(f"{key}: {value}")

    def __call__(self, state: GraphState):
        return self.execute(state)


# ===============LanguageCheckerNode===============#
class LanguageCheckerNode(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "LanguageCheckerNode"
        self.checker_chain = create_language_checker_chain()

    def execute(self, state: GraphState) -> str:
        question = state["question"]
        documents = state["documents"]
        evaluation = self.checker_chain.invoke(
            {"question": question, "documents": documents}
        )

        if evaluation.binary_score == "yes":
            return "english"
        else:
            return "non-English"


# ===============AccentClassifierNode===============#
class AccentClassifierNode(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "AccentClassifierNode"
        self.classifier_chain = create_accent_classifier_chain()

    def execute(self, state: GraphState) -> GraphState:
        question = state["question"]
        documents = state["documents"]
        accent_result = self.classifier_chain.invoke(
            {"question": question, "documents": documents}
        )

        return GraphState(accent_type=accent_result.accent_label)


# ===============AccentGraderNode===============#


class AccentGraderNode(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "AccentGraderNode"
        self.grader_chain = create_accent_grader_chain()

    def execute(self, state: GraphState) -> GraphState:
        question = state["question"]
        documents = state["documents"]
        evaluation = self.grader_chain.invoke(
            {"question": question, "documents": documents}
        )

        return GraphState(accent_grade=evaluation.accent_score)


# ===============RetrievalNode===============#


class RetrievalNode(BaseNode):
    def __init__(self, retriever, **kwargs):
        super().__init__(**kwargs)
        self.name = "RetrievalNode"
        self.retriever = retriever

    def execute(self, state: GraphState) -> GraphState:
        question = state["question"]
        documents = self.retriever.invoke(question)
        return GraphState(documents=documents)


# ===============RetrievalAnswerNode===============#
class RetrievalAnswerNode(BaseNode):
    def __init__(self, retrieval_chain, **kwargs):
        super().__init__(**kwargs)
        self.name = "RetrievalAnswerNode"
        self.retrieval_chain = retrieval_chain

    def execute(self, state: GraphState) -> GraphState:
        question = state["question"]
        documents = state["documents"]
        accent_type = state["accent_type"]
        accent_score = state["accent_grade"]
        answer = self.retrieval_chain.invoke(
            {
                "documents": documents,
                "question": question,
                "accent_type": accent_type,
                "accent_grade": accent_score,
            }
        )
        return GraphState(generation=answer)
