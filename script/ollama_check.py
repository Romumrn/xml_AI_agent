import requests
import logging
from typing import List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class OllamaManager:
    BASE_URL = "http://localhost:11434/api"

    @classmethod
    def check_connection(cls) -> bool:
        """Vérifie si Ollama est accessible"""
        try:
            response = requests.get(f"{cls.BASE_URL}/tags", timeout=10)
            response.raise_for_status()
            return True
        # # if ollama is not currently running, start it
        # import subprocess
        # subprocess.Popen(["ollama","serve"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except requests.exceptions.ConnectionError:
            logger.error("❌ Cannot connect to Ollama. Make sure Ollama is running with 'ollama serve'")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error connecting to Ollama: {e}")
            return False

    @classmethod
    def get_available_models(cls) -> List[str]:
        """Récupère la liste des modèles disponibles dans Ollama"""
        try:
            response = requests.get(f"{cls.BASE_URL}/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to get model list: {e}")
            return []

    @classmethod
    def check_model_availability(cls, model_name: str) -> bool:
        """Vérifie si un modèle spécifique est disponible"""
        available_models = cls.get_available_models()
        
        # Vérification exacte
        if model_name in available_models:
            return True
        
        # Vérification avec variantes (ex: gemma2:9b -> gemma2:9b-instruct)
        model_base = model_name.split(':')[0] if ':' in model_name else model_name
        for available in available_models:
            if available.startswith(model_base):
                return True
        
        return False

    @classmethod
    def validate_models(cls, main_model: str, coord_model: str, attr_model: str) -> bool:
        """Valide que tous les modèles requis sont disponibles"""
        models_to_check = [main_model, coord_model, attr_model]
        all_valid = True
        
        logger.info("🔍 Checking model availability...")
        
        for model in set(models_to_check):  # éviter les doublons
            if cls.check_model_availability(model):
                logger.info(f"✅ Model '{model}' is available")
            else:
                logger.error(f"❌ Model '{model}' is not available")
                all_valid = False
                logger.info(f"   You can install it with: ollama pull {model}")
        
        return all_valid

    @classmethod
    def list_available_models(cls):
        """Affiche tous les modèles disponibles"""
        models = cls.get_available_models()
        if models:
            logger.info("📋 Available models:")
            for model in sorted(models):
                logger.info(f"   - {model}")
        else:
            logger.warning("No models found or unable to connect to Ollama")
