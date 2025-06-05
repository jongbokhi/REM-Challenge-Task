from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain_core.prompts import load_prompt

MODEL_NAME = "gpt-4o-mini"


def create_retrieval_chain(prompt_name="retrieval_chain_prompt", model=MODEL_NAME):
    # Set prompt
    retrieval_prompt = load_prompt(f"prompt/{prompt_name}.yaml")

    # Set llm
    llm = ChatOpenAI(model_name=model, temperature=0)

    # Set chain
    retrieval_chain = (
        {
            "question": itemgetter("question"),
            "documents": itemgetter("documents"),
            "accent_type": itemgetter("accent_type"),
            "accent_grade": itemgetter("accent_grade"),
        }
        | retrieval_prompt
        | llm
        | StrOutputParser()
    )

    return retrieval_chain
