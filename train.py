from dataset import *
from model import *
from utils import *
from config import *
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runtime/logs/')

def train(model, batch_size, optimizer, criterion, n_epochs):
    batch_losses = []
    
    model.train()

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch in tqdm(range(1, n_epochs + 1)):
        # initialize hidden state
        hidden = model.init_hidden(batch_size)
        
        for batch_i, (inputs, labels) in enumerate(train_data_loader, 1):
            n_batches = len(train_data_loader.dataset) // batch_size
            if(batch_i > n_batches):
                break
            loss, hidden = forward_back_prop(model, optimizer, criterion, inputs, labels, hidden)          
            
            # loss 기록
            batch_losses.append(loss)
            writer.add_scalar("Loss/train/seq", np.mean(batch_losses), epoch)
            
            # printing loss stats
            if batch_i % PRINT_EVERY == 0:
                print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                    epoch, n_epochs, np.mean(batch_losses)))
                batch_losses = []

    return model

def forward_back_prop(model, optimizer, criterion, inp, target, hidden):
    
    h = tuple([each.data for each in hidden])
    
    model.zero_grad()
    
    inputs, targets = inp, target
    output, h = model(inputs, h)
    print(output.shape)
    loss = criterion(output, targets.long())
    
    loss.backward()
    # Gradient clipping : 학습 중 Gradient Vanishing 또는 Exploding 방지
    # LSTM은 미분값이 매우 크거나 작아질 수 있음
    # Gradient의 최대 개수 제한
    nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
    
    return loss.item(), h

data_path = 'data\lyric.txt'
d = Dataset(data_path)
lyric2idx, word2idx, idx2word, token_dict = d.load_preprocessed_data() 
train_data_loader = d.batch_data(lyric2idx, SEQUENCE_LENGTH, BATCH_SIZE)
vocab_size = len(word2idx)
output_size = vocab_size
    
model = RNN(vocab_size, output_size, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, dropout = DROPOUT)
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss()    
    
trained_model = train(model, BATCH_SIZE, optimizer, criterion, NUM_EPOCHS)
save_model('./runtime/trained_seq', trained_model)
print('Model Saved') 
writer.flush()   
writer.close()
