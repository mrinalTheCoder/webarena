import os

import google.generativeai as genai


def generate_from_gemini_completion(
    prompt: str,
    model: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    stop_sequences: list[str] | None = None,
) -> str:
    API_KEY = os.environ.get("GEMINI_API_KEY")
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(model)
    config = genai.GenerationConfig(
        max_output_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stop_sequences=stop_sequences,
    )
    return model.generate_content(prompt, generation_config=config).text
