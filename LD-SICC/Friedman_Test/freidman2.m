clc;
clear;
data_acc=csvread('accuracy2.csv');
labels={'LD-SICC','1NN_M_D_D_T_W','SVM_M_D_D_T_W','EDTW ','LBM','TDVM','DSVM','SVM_L_I_N ','SVM_S_I_G','SVM_R_B_F','SVM_D_T_W','1NN_W_D','SMTS','KLD-GMC'};
% alpha=0.1; %显著性水平0.1,0.05或0.01
alpha=0.05; %显著性水平0.1,0.05或0.01
% alpha=0.01; %显著性水平0.1,0.05或0.01
cd=criticaldifference(data_acc,labels,alpha);

