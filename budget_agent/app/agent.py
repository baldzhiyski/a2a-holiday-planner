# agent.py
import json
from typing import Optional

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()


class BudgetOutput(BaseModel):
    flight_cap_eur: float = Field(..., description="Cap for flights in EUR")
    hotel_cap_eur: float = Field(..., description="Cap for hotel in EUR")
    activities_cap_eur: float = Field(..., description="Cap for activities in EUR")
    notes: Optional[str] = Field(None, description="Optional notes")


class BudgetAgent:
    """
    Direct structured-output agent (no LangGraph / no ReAct).
    The model returns a STRICT BudgetOutput JSON.
    """

    SYSTEM = (
        "You are BudgetPolicy. You ONLY help with budget allocation for trips.\n"
        "Inputs: total_budget_eur and passengers.\n"
        "Allocate caps: flights ≈45%, hotel ≈40%, activities ≈15%.\n"
        "For 2+ passengers: slightly increase activities (~+10%) and slightly decrease hotel (~-5%).\n"
        "Return ONLY a JSON object that matches the BudgetOutput schema exactly.\n"
        "Do NOT include any extra text or keys."
    )

    SUPPORTED_CONTENT_TYPES = ["text/plain"]

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        self.llm = ChatOpenAI(model=model, temperature=temperature)

        # Build a small chain: prompt -> structured output
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.SYSTEM),
                (
                    "user",
                    "User text: {user_text}\n\n"
                    "Extract `total_budget_eur` and `passengers` from the user text and produce BudgetOutput.",
                ),
            ]
        )
        self.chain = self.prompt | self.llm.with_structured_output(BudgetOutput)

    def invoke(self, user_text: str) -> BudgetOutput:
        """
        Synchronously generates a BudgetOutput from arbitrary user text.
        Example user_text:
            - 'We have a total budget of 2500 EUR for 3 passengers.'
            - 'total_budget_eur=1800, passengers=2'
        """
        # You can optionally pre-normalize user_text here if you want.
        result: BudgetOutput = self.chain.invoke({"user_text": user_text})
        return result

