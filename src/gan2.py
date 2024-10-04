# Transformer-based Generator
import os
import numpy as np
from pathlib import Path
from datetime import date
import argparse
import shutil

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils import data
import json

import utils.logger as logger


class Dataloader(data.Dataset):
    # TODO: In original paper they have taken disjoint sequences and not a sliding window!
    file_path = Path(__file__).absolute()
    base_dir = file_path.parents[1]
    embeddings_dir = base_dir / 'output'
    raw_data_dir = base_dir / 'data' / 'raw' / 'syllable_level_npy_39'

    def read_file(self, f_path):
        f_data = np.load(f_path, allow_pickle=True)
        cont_attributes = f_data[0][0]
        discrete_attributes = f_data[0][1]
        lyrics = f_data[0][2]
#         lyrics = f_data[0][2][:100]

        # print(type(cont_attributes))
        # print(cont_attributes)
        return cont_attributes, discrete_attributes, lyrics

    def convert_lyrics_to_ix(self, lyrics):
        lyrics_ix = [self.word_to_ix[i] for i in lyrics]
        return lyrics_ix

    def convert_lyrics_to_embeddings(self, lyrics):
        lyrics_embeddings = [self.embeddings_vec[self.word_to_ix[i]] for i in lyrics]
        return lyrics_embeddings

    def generate_ngrams(self, lst, n):
        # Use the zip function to help us generate n-grams
        # Return a list of tuples
        # Each tuple is (word_i-2, word_i-1, word_i)
        ngrams = zip(*[lst[i:] for i in range(n)])
        return [list(ngram) for ngram in ngrams]

    # def convert_lst_tensor_to_tensor(self, lst_tensor):
    #     out_tensor = torch.Tensor(len(lst_tensor), lst_tensor[0].shape[0])
    #     print(out_tensor)
    #     torch.cat(lst_tensor, out = out_tensor)
    #     return out_tensor
    #
    # def convert_lyrics_seq_to_tensor(self, lyrics_seq):
    #     # print(lyrics_seq)
    #     print(lyrics_seq[0])


    def create_melody_seq(self, cont_attr, discrete_attr, lyrics):
        """
        Takes in 3 lists and creates sequences out of it!
        :param cont_attr:
        :param discrete_attr:
        :param lyrics:
        :return:
        """
        lyrics_seq = self.generate_ngrams(lyrics, self.seq_len)
        # self.convert_lyrics_seq_to_tensor(lyrics_seq)
        # print(lyrics_seq)

        # print("Length of cont attributes")
        # print(len(cont_attr))
        cont_attr_seq = self.generate_ngrams(cont_attr, self.seq_len)
        # print(cont_attr_seq)

        discrete_attr = self.generate_ngrams(discrete_attr, self.seq_len)
        # seq = zip(*[lyrics_seq, cont_attr_seq, discrete_attr])
        return lyrics_seq, cont_attr_seq, discrete_attr

    def create_training_data(self):
        # all_seq is a list of all the sequences.
        # This might explode when dealing with the entire data.
        # Might need an alternate way out!!
        # TODO: Check this with entire data
        all_lyrics_seq = []
        all_cont_attr_seq = []
        all_discrete_attr_seq = []
        f_names = self.raw_data_dir.iterdir()
        for i, f_name in enumerate(f_names):
            cont_attr, discrete_attr, lyrics = self.read_file(f_name)
            # print(lyrics)

            # TODO: Remove creating lyrics and the function if not being used below!
            lyrics_ix = self.convert_lyrics_to_ix(lyrics)
            # print(lyrics_ix)
            lyrics_embeddings = self.convert_lyrics_to_embeddings(lyrics)
            # print(type(lyrics_embeddings))

            # TODO: Decide to what to use here. Embeddings directly or just lyrics index
            lyrics_seq, cont_attr_seq, discrete_attr_seq = self.create_melody_seq(cont_attr, discrete_attr, lyrics_embeddings)

            # print("Printing here")
            # # print(len(discrete_attr_seq))
            # print(cont_attr_seq)
            # print(lyrics_seq)
            # print(discrete_attr_seq)

            # print(f_seq)
            all_lyrics_seq.extend(lyrics_seq)
            all_cont_attr_seq.extend(cont_attr_seq)
            all_discrete_attr_seq.extend(discrete_attr_seq)
            # TODO: Remove the break statement.
            break
        return all_lyrics_seq, all_cont_attr_seq, all_discrete_attr_seq

    def __init__(self, embeddings_fname, vocab_fname, seq_len):
        embeddings_vec = torch.load(self.embeddings_dir/ embeddings_fname)
        # TODO: Comment out the line below.
#         embeddings_vec = embeddings_vec[:, :10]
        self.embeddings_vec = embeddings_vec.tolist()
        # print(self.embeddings_vec)
        with open(self.embeddings_dir / vocab_fname, 'r') as fp:
            self.word_to_ix = json.load(fp)
        # print(word_to_ix)

        self.seq_len = seq_len
        lyrics_seq, cont_attr_seq, discrete_attr_seq = self.create_training_data()

        # print(lyrics_seq)
        # print(cont_attr_seq)
        # print(discrete_attr)

        # print("Length of lyrics seq list: {}".format(len(lyrics_seq)))
        # print("Shape of one element: {}".format(len(lyrics_seq[0])))
        # print("An element is: {}".format(lyrics_seq[0]))

        # lyrics_seq_tensor = torch.Tensor(len(lyrics_seq), seq_len, 10)
        # torch.cat(lyrics_seq, out=lyrics_seq_tensor)
        # print(lyrics_seq_tensor[0])
        # self.lyrics_seq = lyrics_seq_tensor

        self.lyrics_seq = torch.Tensor(lyrics_seq)

        self.cont_attr_seq = torch.tensor(cont_attr_seq)

        # print(self.cont_attr_seq.shape)

        self.discrete_attr_seq = torch.tensor(discrete_attr_seq)
        # , ,  = self.create_training_data()

    def __len__(self):
#         print(len(self.lyrics_seq))
        return len(self.lyrics_seq)

    def __getitem__(self, i):
        lyrics_seq = self.lyrics_seq[i]
        cont_val_seq = self.cont_attr_seq[i]
        discrete_val_seq = self.discrete_attr_seq[i]
        #TODO: Add noise shape as a parameter in the class
        noise_seq = torch.rand(lyrics_seq.shape)

        return lyrics_seq, cont_val_seq, discrete_val_seq, noise_seq


class GeneratorTransformer(nn.Module):
    def __init__(self, embed_dim, ff1_out, hidden_dim, out_dim, num_layers=2, num_heads=1):
        super(GeneratorTransformer, self).__init__()

        # Ensure embed_dim is correctly set to 128
        self.embed_dim = embed_dim  # This should be 128
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads

        self.input_ff = nn.Linear(embed_dim * 2, ff1_out)  # Update to match concat_lyrics shape
        encoder_layers = nn.TransformerEncoderLayer(d_model=ff1_out, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.output_ff = nn.Linear(ff1_out, out_dim)

    def forward(self, lyrics, noise):
        # assert lyrics.shape[-1] == self.embed_dim, f"Expected lyrics last dimension to be {self.embed_dim}, got {lyrics.shape[-1]}"
        # assert noise.shape[-1] == self.embed_dim
        concat_lyrics = torch.cat((lyrics, noise), dim=2)
        print("@NOISE" ,lyrics.shape)
        # Ensure the shape of concat_lyrics is (batch_size, seq_len, embed_dim * 2)
        print("concat_lyrics shape:", concat_lyrics.shape,self.embed_dim)
        assert concat_lyrics.shape[-1] == 2 * self.embed_dim, "Concat lyrics must have 2 * embed_dim as the last dimension"
        out1 = F.relu(self.input_ff(concat_lyrics))
        out1 = out1.permute(1, 0, 2)  # (batch_size, seq_len, ff1_out) -> (seq_len, batch_size, ff1_out)
        transformer_out = self.transformer_encoder(out1)
        transformer_out = transformer_out.permute(1, 0, 2)  # (seq_len, batch_size, ff1_out) -> (batch_size, seq_len, ff1_out)
        tag = self.output_ff(transformer_out)
        return tag

class DiscriminatorTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_layers=2, num_heads=1):
        super(DiscriminatorTransformer, self).__init__()

        # Ensure input_dim is correctly set to 131
        self.input_dim = input_dim  # This should be 131
        self.num_heads = num_heads
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        self.head_dim = input_dim // num_heads

        encoder_layers = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.out = nn.Linear(input_dim, out_dim)

    def forward(self, input, lyrics):
        concat_input = torch.cat((input, lyrics), dim=2)
        # Ensure the shape of concat_input is (batch_size, seq_len, input_dim)
        # print("concat_input shape:", concat_input.shape)
        assert concat_input.shape[-1] == self.input_dim, "Concat input must have input_dim as the last dimension"
        concat_input = concat_input.permute(1, 0, 2)  # (batch_size, seq_len, input_dim) -> (seq_len, batch_size, input_dim)
        transformer_out = self.transformer_encoder(concat_input)
        transformer_out = transformer_out.permute(1, 0, 2)  # (seq_len, batch_size, input_dim) -> (batch_size, seq_len, input_dim)
        last_layer_out = transformer_out[:, -1, :]  # Get output of the last timestep
        out = torch.sigmoid(self.out(last_layer_out))
        return out

# Updated LossCompute for Binary Cross Entropy Loss
class LossCompute(object):
    def __init__(self):
        self.criterion = nn.BCELoss()

    def __call__(self, x, y):
        return self.criterion(x, y)
def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = torch.ones(size)
    return data

def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = torch.zeros(size)
    return data
# Training function for conditional GAN
def train_conditional_gan(train_data_iterator, generator, discriminator, optimizer_G, optimizer_D, criterion, start_epoch, epochs, loss_threshold, device, checkpoint_dir, model_dir, save_every, print_every, train_D_steps, train_G_steps):
    for epoch in range(start_epoch, epochs):
        losses_G = []
        losses_D = []

        discriminator.train()
        generator.train()

        # Train discriminator
        total_D_loss = 0
        for num_steps_D, data in enumerate(train_data_iterator):
            lyrics_seq, cont_val_seq, discrete_val_seq, noise_seq = [d.to(device) for d in data]

            optimizer_D.zero_grad()

            fake_G_out = generator(lyrics_seq, noise_seq).detach()
            fake_D_out = discriminator(fake_G_out, lyrics_seq)
            fake_val = zeros_target(fake_D_out.shape).to(device)
            fake_D_loss = criterion(fake_D_out, fake_val)
            fake_D_loss.backward()

            true_D_out = discriminator(discrete_val_seq, lyrics_seq)
            true_val = ones_target(true_D_out.shape).to(device)
            true_D_loss = criterion(true_D_out, true_val)
            true_D_loss.backward()

            optimizer_D.step()
            total_D_loss += (fake_D_loss.item() + true_D_loss.item()) / 2

            if num_steps_D == train_D_steps:
                break

        losses_D.append(total_D_loss)
        print("Loss while training discriminator is: {}".format(total_D_loss))

        # Train Generator
        total_G_loss = 0
        for num_steps_G, data in enumerate(train_data_iterator):
            lyrics_seq, cont_val_seq, discrete_val_seq, noise_seq = [d.to(device) for d in data]

            optimizer_G.zero_grad()

            fake_G_out = generator(lyrics_seq, noise_seq)
            fake_D_out = discriminator(fake_G_out, lyrics_seq)
            true_val = ones_target(fake_D_out.shape).to(device)
            fake_G_loss = criterion(fake_D_out, true_val)
            fake_G_loss.backward()
            optimizer_G.step()

            total_G_loss += fake_G_loss.item()

            if num_steps_G == train_G_steps:
                break

        losses_G.append(total_G_loss)
        print("EPOCH: {} | Loss while training generator is: {}".format(epoch, total_G_loss))

        # Save checkpoints if needed
        if (epoch + 1) % save_every == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
            }
            torch.save(checkpoint, f'{checkpoint_dir}/gan_checkpoint_{epoch + 1}.pth')
            print(f"Checkpoint saved at epoch {epoch + 1}")
            
            checkpoint_path_G = os.path.join(model_dir, f'generator_epoch_{epoch + 1}.pt')
            checkpoint_path_D = os.path.join(model_dir, f'discriminator_epoch_{epoch + 1}.pt')

            torch.save(generator.state_dict(), checkpoint_path_G)
            torch.save(discriminator.state_dict(), checkpoint_path_D)

            print(f"Model checkpoints saved at epoch {epoch + 1}")

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    print("DEVICE: " ,device)
    data_params = {'batch_size': 256, 'shuffle': True, 'num_workers': 6}
    learning_rate_G = 0.0001
    learning_rate_D = 0.0001

    sequence_len = 20
    training_set = Dataloader('2024-10-04_embeddings_vector.pt', '2024-10-04_vocabulary_lookup.json', sequence_len)
    train_data_iterator = data.DataLoader(training_set, **data_params)

    embed_dim = 128
    lyrics_dim = 2 * embed_dim
    ff1_out = 400
    hidden_dim = 400
    generator_out_dim = 3

    discriminator_input_dim = embed_dim + generator_out_dim
    discriminator_out_dim = 1

    generator = GeneratorTransformer(embed_dim, ff1_out, hidden_dim, generator_out_dim)
    discriminator = DiscriminatorTransformer(discriminator_input_dim, hidden_dim, discriminator_out_dim)

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate_G)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate_D)

    criterion = LossCompute()
    start_epoch = 0
    epochs = 1000
    train_D_steps = 1
    train_G_steps = 1
    # print("GENERATOR",generator)
    # print("Discriminator",discriminator)
    
    current_dir = os.getcwd()

# Define the relative path
    relative_path = './data/model_checkpoint'

# Join the current directory with the relative path
    full_path = os.path.abspath(os.path.join(current_dir, relative_path))
    model_path = os.path.abspath(os.path.join(current_dir, './model'))
    print(full_path, model_path)
    train_conditional_gan(train_data_iterator, generator, discriminator, optimizer_G, optimizer_D, criterion, start_epoch, epochs, 'loss_threshold', device, full_path, model_path, 100, 10, train_D_steps, train_G_steps)