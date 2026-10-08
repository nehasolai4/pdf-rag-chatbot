import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def generate_answer(question, context):

    prompt = f"""
You are a helpful PDF question-answering assistant.

Answer the user's question using ONLY the information provided
in the PDF context below.

If the answer cannot be found in the context, say:
"I couldn't find the answer in the uploaded PDF."

PDF CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        input=prompt
    )

    return interaction.output_text