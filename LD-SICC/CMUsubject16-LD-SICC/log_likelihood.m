function log_likelihood = log_likelihood(X,Ma_label,means_label)
%用来得到对数似然ll
%Ma 逆协方差矩阵
%means 均值矩阵
%X_label 子序列的类别
%X 子序列
%X_num 子序列在测试集的编号
%cn 类的个数

ld=log(det(Ma_label));%对数似然ll(Xi,Mai)的一部分logdet(Ma{})
ll=0;

[num_subsequencei_sample,~]=size(X);%子序列X的样本大小The number of subsequence sample
for j=1:num_subsequencei_sample
    x=X(j,:)-means_label;% (x-u)
    ll=-(x*Ma_label)*x'+ld+ll;%第i个子序列对应第k个类的对数似然值
end
log_likelihood=ll;%逐行扫描，得到第i个子序列的预测类别，r为最大对数似然值对应的类别下标
end