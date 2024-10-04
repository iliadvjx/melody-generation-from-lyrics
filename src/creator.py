import os
import numpy as np
import torch
import torch.nn as nn
from conditioned_gan import GeneratorLSTM
from utils import midi_statistics
from utils import utils
from gan2 import GeneratorTransformer
from gensim.models import Word2Vec
syll_model_path = './enc_models/syllEncoding_20190419.bin'
word_model_path = './enc_models/wordLevelEncoder_20190419.bin'


current_dir = os.getcwd()
syllModel = Word2Vec.load(os.path.join(current_dir,syll_model_path))
wordModel = Word2Vec.load(os.path.join(current_dir,word_model_path))

'''
lyrics = [['Must','Must'],['have','have'],['been','been'],['love','love'],
          ['but','but'],['its','its'],['o','over'],['ver','over'],['now','now'],['lay','lay'],['a','a'],
          ['whis','whisper'],['per','whisper'],['on','on'],['my','my'],['pil','pillow'],['low','pillow']]
lyrics = [['Then','Then'],['the','the'],['rain','rainstorm'],['storm','rainstorm'],['came','came'],
          ['ov','over'],['er','over'],['me','me'],['and','and'],['i','i'],['felt','felt'],['my','my'],
          ['spi','spirit'],['rit','spirit'],['break','break']]
lyrics = [['E','Everywhere'],['very','Everywhere'],['where','Everywhere'],['I','I'],['look','look'],
         ['I','I'],['found','found'],['you','you'],['look','looking'],['king','looking'],['back','back']]
'''
lyrics = [['You','You'],['turn','turn'],['my','my'],['nights','nights'],
          ['in','into'],['in','into'],['days','days'],['Lead','Lead'],['me','me'],['mys','mysterious'],['te','mysterious'],
          ['ri','mysterious'],['ous','mysterious'],['ways','ways']]

length_song = len(lyrics)
cond = []

for i in range(20):
    if i < length_song:
        syll2Vec = syllModel.wv[lyrics[i][0]]
        word2Vec = wordModel.wv[lyrics[i][1]]
        cond.append(np.concatenate((syll2Vec,word2Vec)))
    else:
        cond.append(np.concatenate((syll2Vec,word2Vec)))


flattened_cond = []
for x in cond:
    for y in x:
        flattened_cond.append(y)

#/kaggle/working/melody-generation-from-lyrics/model/generator_epoch_100.pt
model_path = os.path.join(current_dir,'./model/generator_epoch_100.pt' ) 
# model_path = os.path.join(current_dir,'./data/model_checkpoint/gan_checkpoint_50.pth' ) 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(model_path, map_location=device)

# Print out the checkpoint contents to find the model parameters
print(checkpoint.keys())  # This will show the keys in the checkpoint dictionary

# If the state_dict is saved directly
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint  # Assuming the checkpoint contains only the state_dict

# Check the dimensions of the saved parameters
# for key, value in state_dict.items():
    # print(f"{key}: {value.shape}")
    
    # Example input
# batch_size = 1
# seq_len = 20
# embed_dim = 128  # This must match the expected embedding dimension

# Generate random lyrics and noise data with the correct shape
# lyrics = torch.randn(batch_size, seq_len, embed_dim)  # shape: (1, 20, 128)
# noise = torch.randn(batch_size, seq_len, embed_dim)   # shape: (1, 20, 128)

# Check input shapes to make sure they align
# print(f"Lyrics shape: {lyrics.shape}, Noise shape: {noise.shape}")
    
    
embed_dim = 128  # Set to the correct embedding dimension, as mentioned
ff1_out = 400    # Example value; set according to your specific needs
hidden_dim = 400 # Example value; set according to your specific model architecture
out_dim = 3      # Output dimension matching your task (e.g., 3 for MIDI generation)
num_layers = 2   # Number of transformer layers
num_heads = 1    # Number of attention heads

# Initialize the model
# generator = GeneratorTransformer(
#     embed_dim=embed_dim, 
#     ff1_out=ff1_out, 
#     hidden_dim=hidden_dim, 
#     out_dim=out_dim, 
#     num_layers=num_layers, 
#     num_heads=num_heads
# )
generator = GeneratorLSTM(
    embed_dim=embed_dim, 
    ff1_out=ff1_out, 
    hidden_dim=hidden_dim, 
    out_dim=out_dim, 
    # num_layers=num_layers, 
    # num_heads=num_heads
)
# print(torch.load(model_path, map_location=device))
generator.load_state_dict(torch.load(model_path, map_location=device))
generator.to(device)
generator.eval()  # Set model to evaluation mode

# Prepare inputs for the model
keep_prob = 1.0  # Dropout keep probability (in PyTorch, dropout is only active during training)
condition = []  # Condition is empty in this example; adjust accordingly
input_dim = 128  # بعد ورودی مورد انتظار مدل
batch_size = 1
seq_len = 20

# Example input data for generating MIDI
input_songdata = torch.tensor(np.random.uniform(size=(batch_size, seq_len, input_dim)), dtype=torch.float32).to(device)
flattened_cond = np.asarray(flattened_cond)  # تبدیل به آرایه NumPy اگر در قالب دیگری باشد
input_metadata = torch.tensor(np.split(flattened_cond, seq_len), dtype=torch.float32).unsqueeze(0).to(device)
print(input_metadata.shape) #torch.Size([1, 20, 20])
print(input_songdata.shape)  # باید (1, 20, 128) باشد
print(input_metadata.shape)  # باید (1, 20, 128) باشد
print("@++++++++++=========+++++++++++@")
projection_layer = nn.Linear(20, embed_dim).to(device=device)  # تبدیل ابعاد از 20 به 128
input_metadata_projected = projection_layer(input_metadata)  # حالا ابعاد (1, 20, 128) دارد
# Disable gradient computation for inference
with torch.no_grad():
    # Forward pass through the generator
    generated_features = generator(input_songdata, input_metadata_projected)
    generated_features = generated_features.cpu().numpy()  # Convert to NumPy array

# Process the generated features
# sample = [x[0, :] for x in generated_features]
sample = generated_features[0]
sample = midi_statistics.tune_song(utils.discretize(sample))
print(generated_features)
print(sample)
# Create MIDI pattern from the discretized data
midi_pattern = utils.create_midi_pattern_from_discretized_data(sample[0:length_song])

# Save the generated MIDI pattern to a file
destination = "test.mid"
midi_pattern.write(destination)

print('done')