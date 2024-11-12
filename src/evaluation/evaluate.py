import re
import os
import torch

MODEL_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../models')

SHORT_TEXTS = [
  "Once upon a time,",
  "In a land far far away,",
  "There was a princess named",
  "She lived in a castle with",
  "She had a pet dragon named",
  "The dragon was very",
  "One day, the princess",
]

LONG_TEXTS = [
  "Once upon a time, in a land far far away, there was a princess named Alice. She lived in a castle with her pet dragon named Bob. The dragon was very friendly. One day, the princess went on a walk in the forest.",
  "Long ago, in a kingdom by the sea, there was a king named Arthur. He lived in a castle with his queen. One day, the king went to",
  "Deep in the heart of the forest, there was a fox named Max. Max was a good fox."
  "There once was a young boy named Dan. Dan liked to run more than anything in the world. One day,",
  "In a small town lived a young girl named Lily. Lily was a"
]

def generate_response(model, tokenizer, text):
    model.eval()
    with torch.no_grad():
      encoded_text = tokenizer.encode(text)
      start_tokens = len(encoded_text)
      encoded_text = torch.tensor(encoded_text).unsqueeze(0)
      output = model.generate(encoded_text, max_new_tokens=20)[0].tolist()
      output = output[start_tokens:]
      decoded_output = tokenizer.decode(output)
      
    return decoded_output

def evaluate_model_outputs(model, tokenizer):
  
  model_dir = f'{MODEL_BASE_DIR}/{model.name}/'
  model_files = os.listdir(model_dir)
  epoch_numbers = [int(re.search(r'epoch_(\d+)', f).group(1)) for f in model_files if re.search(r'epoch_(\d+)', f)]
  most_recent_epoch = max(epoch_numbers)
  model_state_dict = torch.load(f'{MODEL_BASE_DIR}/{model.name}/{model.name}_epoch_{most_recent_epoch}.pt')
  model.load_state_dict(model_state_dict)
  
  print(f"Loaded model {model.name} from epoch {most_recent_epoch}")
  
  print("=" * 100)
  print("Short Texts:")
  print("=" * 100)
  
  for text in SHORT_TEXTS:
    output = generate_response(model, tokenizer, text)
    print(f"{text} [{output}]")
    
  print("=" * 100)
  print("Long Texts:")
  print("=" * 100)
  
  for text in LONG_TEXTS:
    output = generate_response(model, tokenizer, text)
    print(f"{text} [{output}]")