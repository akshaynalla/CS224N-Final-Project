# CS224N-Final-Project

Final Project for CS 224N:

We worked on the default project: Building a question-answering system (IID SQuAD track). Motivated by recent publications (such as "Attention is All You Need,"" "Machine Comprehension Using Match-LSTM and Answer Pointer," and "Convolutional Neural Networks for Sentence Classification"), we decided to extend the baseline BiDAF model with implementations of a character embedding layer, an answer pointer decoder in place of the original output layer, and a self-attention layer immediately after the bidirectional attention flow layer. We experimented with two versions of character embedding layers, and found that back-to-back convolutional layers allowed for better performances. Our implementations dramatically improved learning speed in the training process. Through multiple rounds of training with various hyperparameters, we achieved F1 scores of 64.83 on the dev set and 63.37 on the test set. We anticipate that this work will aid in the continuing development of efficient question answering systems.

Full report can be found on the stanford CS224N website:
http://web.stanford.edu/class/cs224n/reports/final_reports/report196.pdf

