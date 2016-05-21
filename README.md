#run code
python lstm_adagrad.py
# lstm theano code for question answering with freebase
##encode question with lstm encoder 
hidden state size 150, embedding size 150  
question embedding is sum over hiddent states of lstm encoder  
answer embedding is sum over path(relation) embeddings  
##loss funciton
margin ranking loss which seprate correct answer score with incorrect answer score with at least margin 1. 
##optimizier
adagrad which is a parameter learning rate adaption method.

