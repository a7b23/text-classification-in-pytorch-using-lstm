import re
import unicodedata
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import pickle

class wordIndex(object) :
	def __init__(self) :
		self.count = 0
		self.word_to_idx = {}
		self.word_count = {}
	def add_word(self,word) :
		if not word in self.word_to_idx :
			self.word_to_idx[word] = self.count
			self.word_count[word] = 1
			self.count +=1
		else :
			self.word_count[word]+=1

	def add_text(self,text) :
		for word in text.split(' ') :
			self.add_word(word)


def normalizeString(s):
	s = s.lower().strip()
	s = re.sub(r"<br />",r" ",s)
	# s = re.sub(' +',' ',s)
	s = re.sub(r'(\W)(?=\1)', '', s)
	s = re.sub(r"([.!?])", r" \1", s)
	s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
	
	return s

def limitDict(limit,classObj) :
	dict1 = sorted(classObj.word_count.items(),key = lambda t : t[1], reverse = True)
	count = 0
	for x,y in dict1 :
		if count >= limit-1 :
			classObj.word_to_idx[x] = limit
		else :
			classObj.word_to_idx[x] = count	

		count+=1		

vocabLimit = 50000
max_sequence_len = 500
obj1 = wordIndex()

if torch.cuda.is_available():
    device = torch.device("cuda")
    use_cuda = True
else:
    device = torch.device("cpu")
    use_cuda = False
    
f = open('labeledTrainData.tsv').readlines()

print('reading the lines')

for idx,lines in enumerate(f) :
	if not idx == 0 :
		data = lines.split('\t')[2]
		data = normalizeString(data).strip()
		obj1.add_text(data)


print('read all the lines')

limitDict(vocabLimit,obj1)


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
			return (Variable(torch.zeros(1, 1, self.hidden_dim)).cuda(),Variable(torch.zeros(1, 1, self.hidden_dim)).cuda())
		else:
			return (Variable(torch.zeros(1, 1, self.hidden_dim)),Variable(torch.zeros(1, 1, self.hidden_dim)))

if use_cuda:
	model = Model(50,100).cuda()
else:
	model = Model(50,100)

loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 4

torch.save(model.state_dict(), 'model' + str(0)+'.pth')
print('starting training')

for i in range(epochs) :
	avg_loss = 0.0
	for idx,lines in enumerate(f) :
		if not idx == 0 :
			data = lines.split('\t')[2]
			data = normalizeString(data).strip()
			input_data = [obj1.word_to_idx[word] for word in data.split(' ')]
			#print("input data length ", len(input_data))
			if len(input_data) > max_sequence_len :
				input_data = input_data[0:max_sequence_len]

			if use_cuda:
				input_data = Variable(torch.cuda.LongTensor(input_data))
			else:
				input_data = Variable(torch.LongTensor(input_data))

			target = int(lines.split('\t')[1])

			if use_cuda:
				target_data = Variable(torch.cuda.LongTensor([target]))
			else:
				target_data = Variable(torch.LongTensor([target]))

			hidden = model.init_hidden()
			y_pred,_ = model(input_data,hidden)
			model.zero_grad()
			loss = loss_function(y_pred,target_data)
			avg_loss += loss.data[0]
			
			if idx%500 == 0 or idx == 1:
				print('epoch :%d iterations :%d loss :%g'%(i,idx,loss.data[0]))

				
			loss.backward()
			optimizer.step()
	torch.save(model.state_dict(), 'model' + str(i+1)+'.pth')			
	print('the average loss after completion of %d epochs is %g'%((i+1),(avg_loss/len(f))))	

with open('dict.pkl','wb') as f :
	pickle.dump(obj1.word_to_idx,f)