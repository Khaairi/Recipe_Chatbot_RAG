import google.generativeai as genai
from PIL import Image
import json
import io
from config import api_config, app_config

class IngredientDetector:
    def __init__(self):
        genai.configure(api_key=api_config.gemini_key)
        self.model = genai.GenerativeModel(app_config.llm_model)

    def detect_ingredients(self, image_files):
        """
        Analyzes a list of images and returns detected ingredients.
        Args:
            image_files: List of file-like objects (from Streamlit)
        """
        try:
            images = []
            for file in image_files:
                file.seek(0)
                images.append(Image.open(file))
            
            if not images:
                return []

            prompt = (
                "Analyze these images and identify all raw cooking ingredients visible across all photos. "
                "Ignore non-food items. "
                "Combine findings from all images into a single list without duplicates. "
                "Return the result strictly as a JSON list of strings. "
                "Example output: [\"eggs\", \"flour\", \"milk\"]"
            )
        
            response = self.model.generate_content([prompt] + images)
            
            text_response = response.text.strip()
            
            if text_response.startswith("```"):
                text_response = text_response.strip("`").replace("json\n", "").replace("json", "")
            
            print(f"response: {text_response}")
            
            return json.loads(text_response)
            
        except Exception as e:
            print(f"Error identifying ingredients: {e}")
            return []