function [ result,ll ] = predict(Ma,X,tn,cn,means)
%用来得到分类结果标签result和对数似然ll

ld=zeros(cn,1);%对数似然ll(Xi,Mai)的一部分
for i=1:cn
    ld(i,1)=log(det(Ma{i}));
end

result=zeros(tn,1);%对分类结果向量初始化，长度为数据集大小
ll=zeros(tn,cn);%对对数似然矩阵初始化，大小为数据集大小*类别个数
for i=1:tn  %从1至数据集大小循环执行
    data_subsequencei=X{i};%第i个子序列数据
    [num_subsequencei_sample,~]=size(data_subsequencei);%第i个子序列的样本大小The number of subsequence sample
    for j=1:num_subsequencei_sample
        for k=1:cn  %k为当前类别
            x=data_subsequencei(j,:)-means(k,:);% (x-u)
            ll(i,k)=(-(x*Ma{k})*x'+ld(k,1))+ll(i,k);%第i个子序列对应第k个类的对数似然值
        end
    end
%     ll(i,:)=ll(i,:)/num_subsequencei_sample;
    [~,r]=max(ll(i,:));%逐行扫描，得到第i个子序列的预测类别，r为最大对数似然值对应的类别下标
    result(i)=r;%得到完整的数据集的预测类别
end

end

