clc;
clear;
data_acc=csvread('accuracy1.csv');
%data_acc=csvread('1.csv');
labels={'LD-SICC','MLP','Encoder','MCNN','t-LeNet','MCDCNN','Time-CNN','TWIESN'};
alpha=0.05; %������ˮƽ0.1,0.05��0.01
cd=criticaldifference(data_acc,labels,alpha);
disp('Over!');

