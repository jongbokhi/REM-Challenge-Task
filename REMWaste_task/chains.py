from typing import Literal, Annotated
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

MODEL_NAME = "gpt-4o-mini"


# ===============LanguageCheckerChain===============#
class LanguageChecker(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="Given a user question, determine if detected language is English. Return 'yes' if it is English, otherwise return 'no'.",
    )


def create_language_checker_chain():
    # initialize llm
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    structured_llm_router = llm.with_structured_output(LanguageChecker)

    # set PromptTemplate
    system = """
    
    You are an expert in language identification. You are analyzing text that has been transcribed from audio extracted from videos. 
    
    Your task is below:

        1.If the language of the text is clearly English, respond with: 'yes'

        2.If the language is not English or you are unsure, respond with: 'no'

    Respond only with yes or no. Do not explain or add anything else.
    """

    language_checker_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "User input: \n\n {question} \n\n Retrieved transcripts: {documents}",
            ),
        ]
    )

    # build question router
    language_checker = language_checker_prompt | structured_llm_router

    return language_checker


# ===============AccentClassifierChain===============#


class ClassAccent(BaseModel):
    """Classification of the English accent present in the text (e.g., British, American, Australian, etc.)."""

    accent_label: Literal[
        "American",
        "British",
        "Australian",
        "Canadian",
        "South African",
        "German",
        "Indian",
        "Korean",
        "Japanese",
        "Chinese",
        "Thai",
        "Filipino",
        "Unknown",
    ] = Field(
        ..., description="Predicted English accent type from the audio transcript."
    )


def create_accent_classifier_chain():
    # initialize llm
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    structured_llm_classifier = llm.with_structured_output(ClassAccent)

    # Set PromptTemplate
    system = """
    You are an expert in accent classification.
    The following text is a transcript from audio extracted from a video.
    Your task is to classify the type of English accent reflected in the speech.
    Choose from the following categories: American, British, Australian, Canadian, Indian, Other, or Unknown.
    Respond only with the name of the accent category (e.g., "British").
    Do not provide any explanation or additional text.
    """

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "User question: \n\n {question} \n\n Retrieved transcripts: {documents}",
            ),
        ]
    )

    # Build accent classifier chain
    accent_classifier = grade_prompt | structured_llm_classifier
    return accent_classifier


# ===============AccentGraderChain===============#
class GradeAccent(BaseModel):
    """Confidence score representing how strongly the text reflects an English accent."""

    accent_score: Annotated[
        float,
        Field(
            ...,
            ge=0,
            le=100,
            description="Confidence score for English accent presence (range: 0–100%)",
        ),
    ]


def create_accent_grader_chain():
    # initialize llm
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeAccent)

    # Set PromptTemplate
    system = """
    You are an expert grader assessing whether a given text reflects an English accent.  
    The text is transcribed from audio extracted from a video.  
    Your task is to evaluate how strongly the speaker's accent resembles a native or near-native English accent.  
    Give a confidence score between 0 and 100 to indicate the strength of the English accent.  
    Higher scores indicate stronger presence of an English accent.  
    If you are unsure or the accent is clearly not English, assign 0 score.  
    Respond only with a numeric score (e.g., 85.3), without explanation or any other text.
    """

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "User question: \n\n {question} \n\n Retrieved transcripts: {documents}",
            ),
        ]
    )

    # Build accent grader
    accent_grader = grade_prompt | structured_llm_grader
    return accent_grader
