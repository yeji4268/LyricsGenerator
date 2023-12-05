import torch
from utils import * 
from dataset import *
from config import *
from dataset import *
import torch.nn.functional as F

def generate(model, prime_id, int_to_vocab, token_dict, pad_value, predict_len=100):
    model.eval()
    
    current_seq = np.full((1, SEQUENCE_LENGTH), pad_value)
    current_seq[-1][-1] = prime_id
    predicted = [int_to_vocab[prime_id]]
    
    for _ in range(predict_len):
        
        current_seq = torch.LongTensor(current_seq)
        
        hidden = model.init_hidden(current_seq.size(0))
        
        output, _ = model(current_seq, hidden)
        
        p = F.softmax(output, dim=1).data
         
        top_k = 5
        p, top_i = p.topk(top_k)
        top_i = top_i.numpy().squeeze()
        
        p = p.numpy().squeeze()
        word_i = np.random.choice(top_i, p=p/p.sum())
        
        word = int_to_vocab[word_i]
        predicted.append(word)     
        
        current_seq = np.roll(current_seq.cpu(), -1, 1)
        current_seq[-1][-1] = word_i
    
    gen_sentences = ' '.join(predicted)

    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        gen_sentences = gen_sentences.replace(' ' + token.lower(), key)
    gen_sentences = gen_sentences.replace('\n ', '\n')
    gen_sentences = gen_sentences.replace('( ', '(')
    
    
    return gen_sentences

data_path = 'data\lyric.txt'
d = Dataset(data_path)
lyric2idx, word2idx, idx2word, token_dict = d.load_preprocessed_data() 

trained_model = load_model('trained_dropout_0_1.pt')

while True:
    word = input("주제를 입력하세요 : ")
    if len(word) < 2:
        print("두 글자 이상의 주제를 입력해주세요")
        continue
    length = int(input("원하는 가사의 길이를 입력하세요 : "))
    #test_text = range(length)
    #test_data_loader = d.batch_data(test_text, sequence_length = length // 10, batch_size = 10)
    for prime_word in word:
        pad_word = SPECIAL_WORDS['PADDING']
        generated_lyrics = generate(trained_model, word2idx[word], idx2word, token_dict, word2idx[pad_word], length)
        print(generated_lyrics)