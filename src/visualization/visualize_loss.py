import os
import json
from matplotlib import pyplot as plt

RESULTS_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../results')
FIGURES_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../figures')

def visualize_loss(*args, num_epochs=None, title="Losses", xlabel="Step", ylabel="Loss"):
  
  colors = ['c', 'r', 'g', 'm', 'y', 'k']
  
  plt.figure(figsize=(10, 6))
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  
  plt.ylim(0, 9.5)
  
  for i, item in enumerate(args):
    if len(item) == 3:
      data, label, color = item
    else:
      data, label = item
      color = colors[i]
    
    losses = data['losses']
    data_epochs = data['num_epochs']
    epoch_len = len(losses) // (data_epochs + 1)
    if num_epochs is not None:
      losses = losses[:num_epochs * epoch_len]
    
    x, y = zip(*losses)
    plt.plot(x, y, label=label, color=color)
    
    for i in range(epoch_len, len(losses), epoch_len):
      x = losses[i][0]
      plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.5)
    
  plt.legend()
  save_title = title.lower().replace(" ", "_")
  os.makedirs(FIGURES_BASE_DIR, exist_ok=True)
  plt.savefig(f'{FIGURES_BASE_DIR}/{save_title}.png')
  plt.show()

def visualize_loss_from_files(*args, num_epochs=None, title="Losses", xlabel="Step", ylabel="Loss", xlim=None, ylim=None):
  
  new_args = []
  
  for i, item in enumerate(args):
    if len(item) == 3:
      file_name, label, color = item
    else:
      file_name, label = item
      
    with open(f'{RESULTS_BASE_DIR}/{file_name}.json', 'r') as f:
      data = json.load(f)
    
    if len(item) == 3:
      new_args.append((data, label, color))
    else:
      new_args.append((data, label))
    
  visualize_loss(*new_args, num_epochs=num_epochs, title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)