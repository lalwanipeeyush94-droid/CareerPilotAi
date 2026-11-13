
import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Load local .env without overwriting existing environment variables
load_dotenv(override=False)


def get_api_key():
    """
    Get API key from Streamlit Cloud Secrets
    or local .env / environment variables.
    """

    # First priority: Streamlit Cloud Secrets
    try:
        key = st.secrets.get("OPENROUTER_API_KEY")
        if key:
            return key.strip()
    except Exception:
        pass

    # Second priority: Environment variable / local .env
    key = os.getenv("OPENROUTER_API_KEY")

    if key:
        return key.strip()

    raise ValueError(
        "OPENROUTER_API_KEY not found. "
        "Add it to Streamlit Secrets or your local .env file."
    )


# Initialize OpenRouter client
client = OpenAI(
    api_key=get_api_key(),
    base_url="https://openrouter.ai/api/v1",
    timeout=60.0
)


def ask_llm(
    prompt,
    model="nvidia/nemotron-3-ultra-550b-a55b:free"
):
    """
    Generate an AI response using OpenRouter.
    """

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=250
        )

        result = response.choices[0].message.content

        if not result:
            return "Error: Model returned an empty response."

        return result.strip()

    except Exception as e:
        error_message = str(e)
        print(f"OpenRouter LLM Error: {error_message}")

        return f"LLM Error: {error_message}"
