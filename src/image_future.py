import openai, os
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI()

def generate_future_body(img_path: str, goal: str = "athletic, defined shoulders"):
    """
    img_path: ścieżka do JPG wgrywanego przez użytkownika
    returns: URL (lub bytes) wygenerowanego obrazu 1024×1024
    """
    prompt = f"Generate an image of the same person but {goal}, full-body studio photo."
    with open(img_path, "rb") as f:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            n=1,
            image=f  # in-painting na bazie user image
        )
    return response.data[0].url
