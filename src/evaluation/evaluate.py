import re
import os
import torch

MODEL_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../models')

SINGLE_WORD = [
  "Once",
  "a",
  "to",
  "he"
]

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
  "Deep in the heart of the forest, there was a fox named Max. Max was a good fox.",
  "There once was a young boy named Dan. Dan liked to run more than anything in the world. One day,",
  "In a small town lived a young girl named Lily. Lily was a"
]

def generate_response(model, tokenizer, text, beam_search=False):
    model.eval()
    if beam_search:
      with torch.no_grad():
        return model.beam_search_generate(text, tokenizer, max_length=100, beam_width=5, max_ngrams=None, only_return_new_tokens=True)
    
    input_ids = tokenizer.encode(text, return_tensors='pt')
    
    output_sequences = model.generate(
        idx=input_ids,
        max_new_tokens=100,
        temperature=1.0,
        top_k=1
    )
    
    output_sequence = output_sequences[0]
    output_sequence = output_sequence[input_ids.size(1):]
    return tokenizer.decode(output_sequence, skip_special_tokens=True)
  
def evaluate_model_outputs(model, tokenizer, beam_search=False):
  
  model_dir = f'{MODEL_BASE_DIR}/{model.name}/'
  model_files = os.listdir(model_dir)
  epoch_numbers = [int(re.search(r'epoch_(\d+)', f).group(1)) for f in model_files if re.search(r'epoch_(\d+)', f)]
  most_recent_epoch = max(epoch_numbers)
  model_state_dict = torch.load(f'{MODEL_BASE_DIR}/{model.name}/{model.name}_epoch_{most_recent_epoch}.pt', map_location=torch.device('cpu'))
  model.load_state_dict(model_state_dict)
  
  print(f"Loaded model {model.name} from epoch {most_recent_epoch}")
  
  print("=" * 100)
  print("Single Word:")
  print("=" * 100)
  
  for text in SINGLE_WORD:
    output = generate_response(model, tokenizer, text, beam_search)
    print(f"{text} [{output}]")
    print("-" * 100)
  
  print("=" * 100)
  print("Short Texts:")
  print("=" * 100)
  
  for text in SHORT_TEXTS:
    output = generate_response(model, tokenizer, text, beam_search)
    print(f"{text} [{output}]")
    print("-" * 100)
    
  print("=" * 100)
  print("Long Texts:")
  print("=" * 100)
  
  for text in LONG_TEXTS:
    output = generate_response(model, tokenizer, text, beam_search)
    print(f"{text} [{output}]")
    print("-" * 100)