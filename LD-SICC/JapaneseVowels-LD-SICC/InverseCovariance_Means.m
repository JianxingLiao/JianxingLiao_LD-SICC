clc;
clear;
addpath('./data')
load JapaneseVowels_mergeData
mergeData=JapaneseVowels_mergeData;

rho=0.02;
maxIt=100;
tol=1e-6;

classnumber=length(mergeData);%类别个数
attributenumber=size(mergeData(1).data,2);%属性个数
means=[];%保存均值
Theta=cell(1,classnumber);
for i=1:classnumber
    means=[means;mean(mergeData(i).data)];
    S=cov(mergeData(i).data);
    [ Theta{1,i},W{1,i} ] = graphicalLasso( S, rho, maxIt, tol );    
end
JapaneseVowels_means=means;
save JapaneseVowels_means.mat JapaneseVowels_means

JapaneseVowels_inverse1=Theta;
save JapaneseVowels_inverse1.mat JapaneseVowels_inverse1

disp('Over!');