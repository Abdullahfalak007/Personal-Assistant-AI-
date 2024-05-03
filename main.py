from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class PersonalAssistant:
    def __init__(self):
        # Load pre-trained GPT-2 model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()  # Set the model to evaluation mode

    def chat(self, input_text):
        # Tokenize input text
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        
        # Generate a response from the model
        with torch.no_grad():
            output = self.model.generate(input_ids, max_length=150)
        
        # Decode the generated tokens to a string
        response_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response_text

# Create an instance of the assistant
assistant = PersonalAssistant()

# Example usage
input_text = "Hello, how are you today?"
response = assistant.chat(input_text)
print(response)
