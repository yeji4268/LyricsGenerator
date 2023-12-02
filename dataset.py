import os
import pickle
import torch
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

SPECIAL_WORDS = {'PADDING': '<PAD>'}

class Vocab_Dict(object):
    def __init__(self, filename):
        self.make_dict(filename)
    
    # 데이터 불러오기
    def load_data(self, filename):
        input_file = os.path.join(filename)
        with open(input_file, "r", encoding = 'UTF-8') as f:
            data = f.read()
            
        # 불러온 데이터의 정보 출력
        print(f'[입력 데이터 정보]')
        print(f'단어 개수 : {len({word: None for word in data.split()})}')
        lines = data.split('\n')
        print(f"가사 데이터 사이즈(줄) : {len(lines)}")
        word_per_line = [len(line.split()) for line in lines]
        print(f"가사 한 줄 당 평균 단어 개수 : {round(np.mean(word_per_line), 2)}")
        return data
    
    # 가사로 단어 사전 만들기
    def make_dict(self, filename):
        # 데이터 불러오기
        text = self.load_data(filename)
        
        # 테스트를 위해 80줄 남겨 놓기
        text = text[81:]

        # 문장 부호 토큰화 : '?' --> QUESTION_MARK
        token_dict = self.make_token_dict()
        for key, token in token_dict.items():
            text = text.replace(key, ' {} '.format(token))

        text = text.lower()
        text = text.split()

        # 인덱스 사전 만들기 
        word2idx, idx2word = self.make_index_dict(text + list(SPECIAL_WORDS.values()))
        
        # 가사를 index 형태로 바꾸기
        lyric2idx = [word2idx[word] for word in text]
        
        # 전처리 된 데이터를 pickle에 dump
        self.to_pickle(lyric2idx, word2idx, idx2word, token_dict)
    
    # 데이터 to pickle   
    def to_pickle(self, lyric2idx, word2idx, idx2word, token_dict):
        pickle.dump((lyric2idx, word2idx, idx2word, token_dict), open('runtime/preprocessed.pkl', 'wb')) 
    
    # 인덱스 사전 만들기
    def make_index_dict(self, text):
        # 인덱스와 가사 속 단어를 매핑
        word_count = Counter(text)
        sorted_vocab = sorted(word_count, key = word_count.get, reverse=True)
        # {인덱스 : '단어'} 형태
        idx2word = {idx: word for idx, word in enumerate(sorted_vocab)}
        # {'단어' : 인덱스} 형태
        word2idx = {word:idx for idx, word in idx2word.items()}
        
        return (word2idx, idx2word)

    # 문장 부호 토큰 사전 만들기
    def make_token_dict(self):
        token = dict()
        token['.'] = '<PERIOD>'
        token[','] = '<COMMA>'
        token['"'] = 'QUOTATION_MARK'
        token[';'] = 'SEMICOLON'
        token['!'] = 'EXCLAIMATION_MARK'
        token['?'] = 'QUESTION_MARK'
        token['('] = 'LEFT_PAREN'
        token[')'] = 'RIGHT_PAREN'
        token['-'] = 'QUESTION_MARK'
        token['\n'] = 'NEW_LINE'
        return token

class Dataset(object):
    def __init__(self, filename):
        self.vocab_dict = Vocab_Dict(filename)
  
    # 전처리된 pickle 형태 데이터 불러오기
    def load_preprocessed_data(self):
        return pickle.load(open('runtime/preprocessed.pkl', mode='rb'))

    # 데이터 배치 처리
    def batch_data(self, words, sequence_length, batch_size):
        n_batches = len(words) // batch_size
        x, y = [], []
        words = words[:n_batches * batch_size]
        
        for idx in range(0, len(words) - sequence_length):
            end_index = idx + sequence_length        
            x_batch = words[idx : idx + sequence_length]
            x.append(x_batch)
            y_batch = words[end_index]
            y.append(y_batch)
        
        data = TensorDataset(torch.from_numpy(np.asarray(x)), torch.from_numpy(np.asarray(y)))
        data_loader = DataLoader(data, shuffle = True, batch_size = batch_size)
            
        return data_loader
