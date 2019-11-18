clc
clear
close all
tic
addpath('./PrecisionRecall')
disp('Loading data...');

load KickvsPunch_mergeData
load KickvsPunch_TRAIN_X
load KickvsPunch_TRAIN_Y
load KickvsPunch_TEST_X
load KickvsPunch_TEST_Y

mergeData=KickvsPunch_mergeData;%每一类所有的子序列集合
TRAIN_X=KickvsPunch_TRAIN_X;%TRAIN_X是训练集样本
TRAIN_Y=KickvsPunch_TRAIN_Y;%TRAIN_Y是训练集标签
TEST_X=KickvsPunch_TEST_X;%TEST_X是测试集样本
TEST_Y=KickvsPunch_TEST_Y;%TEST_Y是测试集标签

%%---设置GraphicalLasso算法的参数---%%
lambda=0.4;
maxIt=200;
tol=0.1;

%%---设置LogDet散度优化算法的参数---%%
eta=0.1;
fac=eta*0.2;
rho=110;
%%---获得均值矩阵-------------  ---------------%%
%%---并通过GraphicalLasso算法得到稀疏逆协方差--%%
classnumber=length(mergeData);%类别个数
attributenumber=size(mergeData(1).data,2);%属性个数
KickvsPunch_means=[];%保存均值
KickvsPunch_inverse=cell(1,classnumber);%保存稀疏逆协方差
for i=1:classnumber
    KickvsPunch_means=[ KickvsPunch_means;mean(mergeData(i).data) ];
end
save KickvsPunch_means.mat KickvsPunch_means
disp('means Over!');
for i=1:classnumber
    S=cov(mergeData(i).data);
%         [ KickvsPunch_inverse{1,i},W{1,i} ] = graphicalLasso( S, lambda, maxIt, tol );
    [ KickvsPunch_inverse{1,i},W{1,i} ] = glasso( S, lambda );
end
save KickvsPunch_inverse.mat KickvsPunch_inverse
disp('inverse Over!');
%---通过LogDet散度进一步优化代表每个类别的稀疏逆协方差---%%
Ma=KickvsPunch_inverse;%保存了每个类别的稀疏逆协方差
Me=Ma;%保存上一次迭代的逆协方差矩阵
saveresults=[];%保存最终的结果
means=KickvsPunch_means; %means矩阵，数据集类别*维度
train_tn=size(TRAIN_X,2);%训练集子序列的数目
test_tn=size(TEST_X,2);%测试集子序列的数目
cn=classnumber;%类的个数

[result,ll]=predict(Ma,TRAIN_X,train_tn,cn,means);%未进一步训练时训练集得到的分类结果及对数似然结果
[~,train_accuracy,~,train_precision,train_recall,train_F1,~,~,~]=compute_accuracy_F (TRAIN_Y,result,cn);%未进一步训练时的训练集分类准确度
fprintf('Train-accuracy: %f \n', train_accuracy);
fprintf('------\n');
re2=predict(Ma,TEST_X,test_tn,cn,means);%未进一步训练时测试集得到的分类结果，测试集的ll未赋值得到
[~,test_accuracy,~,test_precision,test_recall,test_F1,~,~,~]=compute_accuracy_F (TEST_Y,re2,cn);
fprintf('Test-accuracy: %f \n', test_accuracy);
fprintf('-----------------------------------------\n');
test_accuracy_old=test_accuracy;%保存上一次求解的准确率

%----------训练部分--------------%
for kk=1:2%总的循环次数
    for i=1:train_tn %从1至训练样本大小循环执行
        li=ll(i,:); %得到第i个训练样本所有类别的ll对数似然向量
        [lisort,I] = sort(li); %升序，lisort代表排序后的li,I代表排序后的元素对应在li中的下标
        if (result(i)==TRAIN_Y(i)&&(lisort(cn)-lisort(cn-1))<rho)%第一种情况，预测的类和最接近这个类的对数似然大小差距小于rho
            X= TRAIN_X{i}; %X代表训练样本中第i个子序列
            [m,~]=size(X); %m代表第i个子序列的样本大小
            k=TRAIN_Y(i); %第i个训练样本的类别，k属于1到cn
            X1=zeros(size(X));
            for j=1:m %最内层循环，
                X1(j,:)=X(j,:)-means(k,:); %对数似然求解的一部分
            end
            X1=X1'; % 变成了正常的列向量
            %--------------迭代更新-----------------%
            IX=eye(size(X1,2));
            M=Ma{k};%第k个类别的逆协方差
            M=(1+m*eta)*(M - eta*M*X1*(IX + eta*X1'*M*X1)^(-1)*X1'*M);%更新第一个逆协方差
            log_likelihood_old=ll(i,k);
            log_likelihood_new=log_likelihood(TRAIN_X{i},M,means(k,:));
            if(log_likelihood_old<log_likelihood_new)
                Ma{k}=M;
            end
            k1=I(cn-1);
            X2=zeros(size(X));
            for j=1:m
                X2(j,:)=X(j,:)-means(k1,:);
            end
            X2=X2';
            M1=Ma{k1};
            M1=(1-m*fac)*(M1 + fac*M1*X2*(IX - fac*X2'*M1*X2)^(-1)*X2'*M1);%更新第二个逆协方差
            log_likelihood_old=ll(i,k1);
            log_likelihood_new=log_likelihood(TRAIN_X{i},M1,means(k1,:));
            if(log_likelihood_old>log_likelihood_new)
                Ma{k1}=M1;
            end
        elseif(result(i)~=TRAIN_Y(i))%第二种情况，预测错误
            k=TRAIN_Y(i);%正确的标签
            M=Ma{k};
            X= TRAIN_X{i};
            [m,~]=size(X);
            X1=zeros(size(X));
            for j=1:m
                X1(j,:)=X(j,:)-means(k,:);
            end
            X1=X1';
            
            %--------------交替迭代更新-----------------%
            
            IX=eye(size(X1,2));
            M=(1+m*eta)*(M - eta*M*X1*(IX + eta*X1'*M*X1)^(-1)*X1'*M);
            log_likelihood_old=ll(i,k);
            log_likelihood_new=log_likelihood(TRAIN_X{i},M,means(k,:));
            if(log_likelihood_old<log_likelihood_new)
                Ma{k}=M;
            end
            k1=result(i);%错误的类标签
            X2=zeros(size(X));
            for j=1:m
                X2(j,:)=X(j,:)-means(k1,:);
            end
            X2=X2';
            M1=Ma{k1};
            M1=(1-m*fac)*(M1 + fac*M1*X2*(IX - fac*X2'*M1*X2)^(-1)*X2'*M1);
            log_likelihood_old=ll(i,k1);
            log_likelihood_new=log_likelihood(TRAIN_X{i},M1,means(k1,:));
            if(log_likelihood_old>log_likelihood_new)
                Ma{k1}=M1;
            end
        end
    end
    
    [result,ll]=predict( Ma,TRAIN_X,train_tn,cn,means);%分类函数
    [~,train_accuracy,~,train_precision,train_recall,train_F1,~,~,~]=compute_accuracy_F (TRAIN_Y,result,cn);
    fprintf('Train-accuracy: %f \n', train_accuracy);
    fprintf('------%d\n',kk);
    re2=predict( Ma,TEST_X,test_tn,cn,means);
    [~,test_accuracy,~,test_precision,test_recall,test_F1,~,~,~]=compute_accuracy_F (TEST_Y,re2,cn);
    fprintf('Test-accuracy: %f \n', test_accuracy);
    fprintf('------------------------\n');
    saveresults=[saveresults,test_accuracy];
end
[max_test_accuracy,p]=max(saveresults);
Position = find(saveresults==max_test_accuracy);
fprintf('有最好分类准确率的循环次数：');
disp(Position);
fprintf('max_test_accuracy: %f \n', max_test_accuracy);
toc;