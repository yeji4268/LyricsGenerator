from dataset import *
from model import *
from utils import *
from config import *
from tqdm import tqdm

def train(model, batch_size, optimizer, criterion, n_epochs):
    batch_losses = []
    
    model.train()

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch in tqdm(range(1, n_epochs + 1)):
        # initialize hidden state
        hidden = model.init_hidden(batch_size)
        
        for batch_i, (inputs, labels) in enumerate(train_data_loader, 1):
            # make sure you iterate over completely full batches, only
            n_batches = len(train_data_loader.dataset) // batch_size
            if(batch_i > n_batches):
                break
            # forward, back prop
            loss, hidden = forward_back_prop(model, optimizer, criterion, inputs, labels, hidden)          
            # loss 기록
            batch_losses.append(loss)

            # printing loss stats
            if batch_i % PRINT_EVERY == 0:
                print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                    epoch, n_epochs, np.mean(batch_losses)))
                batch_losses = []

    return model

def forward_back_prop(model, optimizer, criterion, inp, target, hidden):
    """
    Forward and backward propagation on the neural network
    :param decoder: The PyTorch Module that holds the neural network
    :param decoder_optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """
    
    h = tuple([each.data for each in hidden])
    
    model.zero_grad()
    

    inputs, targets = inp, target
    
    output, h = model(inputs, h)
    
    loss = criterion(output, targets.long())
    
    # perform backpropagation and optimization
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()

    # return the loss over a batch and the hidden state produced by our model
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
save_model('./runtime/trained', trained_model)
print('Model Saved')    
