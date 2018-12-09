import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import pickle
import re
import numpy as np

def normalizeString(s):
	s = s.lower().strip()
	s = re.sub(r"<br />",r" ",s)
	s = re.sub(r'(\W)(?=\1)', '', s)
	s = re.sub(r"([.!?])", r" \1", s)
	s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
	
	return s

class Model(torch.nn.Module) :
	def __init__(self,embedding_dim,hidden_dim) :
		super(Model,self).__init__()
		self.hidden_dim = hidden_dim
		self.embeddings = nn.Embedding(vocabLimit+1, embedding_dim)
		self.lstm = nn.LSTM(embedding_dim,hidden_dim)
		self.linearOut = nn.Linear(hidden_dim,2)
	def forward(self,inputs,hidden) :
		x = self.embeddings(inputs).view(len(inputs),1,-1)
		lstm_out,lstm_h = self.lstm(x,hidden)
		x = lstm_out[-1]
		x = self.linearOut(x)
		x = F.log_softmax(x)
		return x,lstm_h
	def init_hidden(self) :
		if use_cuda:
			return (Variable(torch.zeros(1, 1, self.hidden_dim).cuda()),Variable(torch.zeros(1, 1, self.hidden_dim)).cuda())
		else:
			return (Variable(torch.zeros(1, 1, self.hidden_dim)),Variable(torch.zeros(1, 1, self.hidden_dim)))


if torch.cuda.is_available():
    device = torch.device("cuda")
    use_cuda = True
else:
    device = torch.device("cpu")
    use_cuda = False
    

vocabLimit = 50000
max_sequence_len = 500
if use_cuda:
	model = Model(50,100).cuda()
else:
	model = Model(50, 100)



with open('dict.pkl','rb') as f :
	word_dict = pickle.load(f)

f = open('testData.tsv').readlines()

model.load_state_dict(torch.load('model3.pth'))

f1 = open('submission.csv','w')

f1.write('"id","sentiment"'+'\n')

for idx,lines in enumerate(f) :
	if not idx == 0 :
		data = normalizeString(lines.split('\t')[1]).strip()
		input_data = []
		for word in data.split(' ') :
			if not word in word_dict :
				input_data.append(vocabLimit)
			else :
				input_data.append(word_dict[word])	
		if len(input_data) > max_sequence_len :
				input_data = input_data[0:max_sequence_len]

		if use_cuda:
			input_data = Variable(torch.cuda.LongTensor(input_data))
		else:
			input_data = Variable(torch.LongTensor(input_data))
		
		hidden = model.init_hidden()
		y_pred,_ = model(input_data,hidden)
		pred1 = y_pred.data.max(1)[1].cpu().numpy()
		#print(pred1)
		f1.write(lines.split('\t')[0]+','+str(pred1[0])+'\n')
		
f1.close()	