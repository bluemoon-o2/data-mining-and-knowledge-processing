import torch
from transformers import AutoTokenizer, AutoModel
from model import MambaClassifier
from config import Config

def test_model():
    print("Testing Mamba Model Setup...")
    config = Config()
    
    # 1. Test Tokenizer
    try:
        print(f"Loading Tokenizer: {config.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        text = "这是一个测试句子。"
        encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=32)
        print("Tokenizer output shape:", encoded['input_ids'].shape)
    except Exception as e:
        print(f"Tokenizer loading failed: {e}")
        return

    # 2. Test Model
    try:
        print(f"Loading Model: {config.model_name}")
        model = MambaClassifier(config.model_name, config.num_classes, variant="default")
        
        # Dummy Input
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        # Forward pass
        with torch.no_grad():
            output = model(input_ids, attention_mask)
        
        print("Model output shape:", output.shape) # Should be (1, 2)
        print("Test Passed!")
    except Exception as e:
        print(f"Model loading/forward failed: {e}")
        print("Make sure you have transformers>=4.39.0 installed.")

if __name__ == "__main__":
    test_model()
