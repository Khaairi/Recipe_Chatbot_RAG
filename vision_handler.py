import google.generativeai as genai
from PIL import Image
import json
import io
from config import api_config, app_config
import base64
import requests

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
        
class IngredientDetectorLlava:
    def __init__(self):
        self.model = app_config.vision_model
        self.api_url = api_config.ollama_url

    def _image_to_base64(self, image_file):
        """Convert an uploaded file (or bytes) to base64 string for Ollama."""
        image_file.seek(0)
        return base64.b64encode(image_file.read()).decode('utf-8')
    
    def detect_ingredients(self, image_files):
        """
        Analyzes a list of images LOCALLY using Ollama (Llava/Moondream).
        """
        all_ingredients = set()

        try:
            for img_file in image_files:
                b64_image = self._image_to_base64(img_file)
                
                prompt = (
                    "Look at this cooking ingredient image. "
                    "List ONLY the food ingredients you see. "
                    "Format the output as a simple comma-separated list. "
                    "Do not write complete sentences. "
                    "output ONLY ONE ingredient"
                    "Example: eggs"
                )

                payload = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [b64_image]
                        }
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.1
                    }
                }

                response = requests.post(self.api_url, json=payload, timeout=120)
                response.raise_for_status()
                
                result_json = response.json()
                content = result_json["message"]["content"]
                
                print(f"content: {content}")
                items = [item.strip().lower() for item in content.replace('\n', ',').split(',')]
                
                for item in items:
                    if item and len(item) > 2: # Filter out empty strings or noise
                        if "image" not in item and "contain" not in item:
                            all_ingredients.add(item.title())

            return list(all_ingredients)

        except requests.exceptions.ConnectionError:
            print("❌ Error: Could not connect to Ollama. Make sure 'ollama serve' is running.")
            return []
        except Exception as e:
            print(f"❌ Vision Error: {e}")
            return []