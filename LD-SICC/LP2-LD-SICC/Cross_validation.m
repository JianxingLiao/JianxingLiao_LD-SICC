clc;
clear;
addpath('./data')
load LP2_TRAIN_X
load LP2_TRAIN_Y
load LP2_TEST_X
load LP2_TEST_Y
load LP2_inverse;
load LP2_means;

TRAIN_X1=LP2_TRAIN_X;%Y文件是X文件的标签
TRAIN_Y1=LP2_TRAIN_Y;
TEST_X1=LP2_TEST_X;
TEST_Y1=LP2_TEST_Y;
means=LP2_means; %均值矩阵，数据集类别*维度

datas=[TRAIN_X1,TEST_X1]';
labels=[TRAIN_Y1,TEST_Y1]';

eta=0.001;
fac=eta*0.2;
rho=100;

Ma=LP2_inverse;
Me=LP2_inverse;

cn=size(Ma,2);%类的个数

k_fold=5;%预将数据分成十份
sum_accuracy_LogDet = 0;
final_saveResults=[];

[m,n] = size(datas);
indices = crossvalind('Kfold',m,k_fold);
TRAIN_X={};
TRAIN_Y=[];
TEST_X={};
TEST_Y=[];
for k_i=1:k_fold
    fprintf('第%d次交叉!\n',k_i);
    saveResults=[];
    %%%%--k_fold交叉验证划分------------------------------------%%%
    test_indic = (indices == k_i);
    train_indic = ~test_indic;
    TRAIN_X = datas(train_indic,:)';%找出训练数据与标签
    TRAIN_Y = labels(train_indic,:)';
    TEST_X = datas(test_indic,:)';%找出测试数据与标签
    TEST_Y = labels(test_indic,:)';
    train_tn=size(TRAIN_X,2);%训练集子序列的数目
    test_tn=size(TEST_X,2);%测试集子序列的数目
    %%%%--初始准确度--------------------------------------------%%%
    [result,ll]=predict(Ma,TRAIN_X,train_tn,cn,means);%未进一步训练时训练集得到的分类结果及对数似然结果
    [~,train_accuracy,~,train_precision,train_recall,train_F1,~,~,~]=compute_accuracy_F (TRAIN_Y,result,cn);%未进一步训练时的训练集分类准确度
    fprintf('Train-accuracy: %f \n', train_accuracy);
    fprintf('------\n');
    re2=predict(Ma,TEST_X,test_tn,cn,means);%未进一步训练时测试集得到的分类结果，测试集的ll未赋值得到
    [~,test_accuracy,~,test_precision,test_recall,test_F1,~,~,~]=compute_accuracy_F (TEST_Y,re2,cn);
    fprintf('Test-accuracy: %f \n', test_accuracy);
    saveResults=[saveResults,test_accuracy];
    fprintf('-----------------------------------------\n');
    %%%%--LogDet优化调整逆协方差--------------------------------%%%
    for kk=1:20%总的循环次数
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
                %             fprintf('第一种情况，%d_%d log_likelihood_old:%f \n',kk,i,log_likelihood_old);
                %             fprintf('第一种情况，%d_%d log_likelihood_new:%f \n',kk,i,log_likelihood_new);
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
        %计算中介中心性得分
        %         Mee=Ma;
        %         for i=1:cn
        %             [bcc{i},EE{i}] = betweenness_centrality(sparse(abs(Mee{i})));
        %         end
        [result,ll]=predict( Ma,TRAIN_X,train_tn,cn,means);%分类函数
        [~,train_accuracy,~,train_precision,train_recall,train_F1,~,~,~]=compute_accuracy_F (TRAIN_Y,result,cn);
        fprintf('Train-accuracy: %f \n', train_accuracy);
        %     fprintf('Train-precision: %f \n', train_precision);
        %     fprintf('Train-recall: %f \n', train_recall);
        %     fprintf('Train-F1_score: %f \n', train_F1);
        fprintf('------%d_%d\n',k_i,kk);
        
        re2=predict( Ma,TEST_X,test_tn,cn,means);
        [~,test_accuracy,~,test_precision,test_recall,test_F1,~,~,~]=compute_accuracy_F (TEST_Y,re2,cn);
        fprintf('Test-accuracy: %f \n', test_accuracy);
        saveResults=[saveResults,test_accuracy];
        %     fprintf('Test-precision: %f \n', test_precision);
        %     fprintf('Test-recall: %f \n', test_recall);
        %     fprintf('Test-F1_score: %f \n', test_F1);
        fprintf('------------------------\n');
    end
    
    [max_test_accuracy,p]=max(saveResults);
    final_saveResults=[final_saveResults,max_test_accuracy];
    fprintf('max_test_accuracy: %f \n', max_test_accuracy);
    sum_accuracy_LogDet = sum_accuracy_LogDet + max_test_accuracy;
end
mean_sum_accuracy_LogDet=sum_accuracy_LogDet/k_fold;
fprintf('%d交叉验证的平均准确率：',k_fold);
disp( mean_sum_accuracy_LogDet );
[final_max_test_accuracy,final_p]=max(final_saveResults);
Position = find(final_saveResults==final_max_test_accuracy);
fprintf('第%d次具有最好的交叉验证测试准确率： %f \n',final_p,final_max_test_accuracy);
fprintf('Over!\n');

