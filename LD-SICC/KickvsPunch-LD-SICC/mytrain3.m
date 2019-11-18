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

mergeData=KickvsPunch_mergeData;%ÿһ�����е������м���
TRAIN_X=KickvsPunch_TRAIN_X;%TRAIN_X��ѵ��������
TRAIN_Y=KickvsPunch_TRAIN_Y;%TRAIN_Y��ѵ������ǩ
TEST_X=KickvsPunch_TEST_X;%TEST_X�ǲ��Լ�����
TEST_Y=KickvsPunch_TEST_Y;%TEST_Y�ǲ��Լ���ǩ

%%---����GraphicalLasso�㷨�Ĳ���---%%
lambda=0.4;
maxIt=200;
tol=0.1;

%%---����LogDetɢ���Ż��㷨�Ĳ���---%%
eta=0.1;
fac=eta*0.2;
rho=110;
%%---��þ�ֵ����-------------  ---------------%%
%%---��ͨ��GraphicalLasso�㷨�õ�ϡ����Э����--%%
classnumber=length(mergeData);%������
attributenumber=size(mergeData(1).data,2);%���Ը���
KickvsPunch_means=[];%�����ֵ
KickvsPunch_inverse=cell(1,classnumber);%����ϡ����Э����
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
%---ͨ��LogDetɢ�Ƚ�һ���Ż�����ÿ������ϡ����Э����---%%
Ma=KickvsPunch_inverse;%������ÿ������ϡ����Э����
Me=Ma;%������һ�ε�������Э�������
saveresults=[];%�������յĽ��
means=KickvsPunch_means; %means�������ݼ����*ά��
train_tn=size(TRAIN_X,2);%ѵ���������е���Ŀ
test_tn=size(TEST_X,2);%���Լ������е���Ŀ
cn=classnumber;%��ĸ���

[result,ll]=predict(Ma,TRAIN_X,train_tn,cn,means);%δ��һ��ѵ��ʱѵ�����õ��ķ�������������Ȼ���
[~,train_accuracy,~,train_precision,train_recall,train_F1,~,~,~]=compute_accuracy_F (TRAIN_Y,result,cn);%δ��һ��ѵ��ʱ��ѵ��������׼ȷ��
fprintf('Train-accuracy: %f \n', train_accuracy);
fprintf('------\n');
re2=predict(Ma,TEST_X,test_tn,cn,means);%δ��һ��ѵ��ʱ���Լ��õ��ķ����������Լ���llδ��ֵ�õ�
[~,test_accuracy,~,test_precision,test_recall,test_F1,~,~,~]=compute_accuracy_F (TEST_Y,re2,cn);
fprintf('Test-accuracy: %f \n', test_accuracy);
fprintf('-----------------------------------------\n');
test_accuracy_old=test_accuracy;%������һ������׼ȷ��

%----------ѵ������--------------%
for kk=1:2%�ܵ�ѭ������
    for i=1:train_tn %��1��ѵ��������Сѭ��ִ��
        li=ll(i,:); %�õ���i��ѵ��������������ll������Ȼ����
        [lisort,I] = sort(li); %����lisort����������li,I����������Ԫ�ض�Ӧ��li�е��±�
        if (result(i)==TRAIN_Y(i)&&(lisort(cn)-lisort(cn-1))<rho)%��һ�������Ԥ��������ӽ������Ķ�����Ȼ��С���С��rho
            X= TRAIN_X{i}; %X����ѵ�������е�i��������
            [m,~]=size(X); %m�����i�������е�������С
            k=TRAIN_Y(i); %��i��ѵ�����������k����1��cn
            X1=zeros(size(X));
            for j=1:m %���ڲ�ѭ����
                X1(j,:)=X(j,:)-means(k,:); %������Ȼ����һ����
            end
            X1=X1'; % �����������������
            %--------------��������-----------------%
            IX=eye(size(X1,2));
            M=Ma{k};%��k��������Э����
            M=(1+m*eta)*(M - eta*M*X1*(IX + eta*X1'*M*X1)^(-1)*X1'*M);%���µ�һ����Э����
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
            M1=(1-m*fac)*(M1 + fac*M1*X2*(IX - fac*X2'*M1*X2)^(-1)*X2'*M1);%���µڶ�����Э����
            log_likelihood_old=ll(i,k1);
            log_likelihood_new=log_likelihood(TRAIN_X{i},M1,means(k1,:));
            if(log_likelihood_old>log_likelihood_new)
                Ma{k1}=M1;
            end
        elseif(result(i)~=TRAIN_Y(i))%�ڶ��������Ԥ�����
            k=TRAIN_Y(i);%��ȷ�ı�ǩ
            M=Ma{k};
            X= TRAIN_X{i};
            [m,~]=size(X);
            X1=zeros(size(X));
            for j=1:m
                X1(j,:)=X(j,:)-means(k,:);
            end
            X1=X1';
            
            %--------------�����������-----------------%
            
            IX=eye(size(X1,2));
            M=(1+m*eta)*(M - eta*M*X1*(IX + eta*X1'*M*X1)^(-1)*X1'*M);
            log_likelihood_old=ll(i,k);
            log_likelihood_new=log_likelihood(TRAIN_X{i},M,means(k,:));
            if(log_likelihood_old<log_likelihood_new)
                Ma{k}=M;
            end
            k1=result(i);%��������ǩ
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
    
    [result,ll]=predict( Ma,TRAIN_X,train_tn,cn,means);%���ຯ��
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
fprintf('����÷���׼ȷ�ʵ�ѭ��������');
disp(Position);
fprintf('max_test_accuracy: %f \n', max_test_accuracy);
toc;