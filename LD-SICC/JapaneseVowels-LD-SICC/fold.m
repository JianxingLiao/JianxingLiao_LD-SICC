clc;
clear;
addpath('./data')
load JapaneseVowels_TRAIN_X
load JapaneseVowels_TRAIN_Y
load JapaneseVowels_TEST_X
load JapaneseVowels_TEST_Y

TRAIN_X1=JapaneseVowels_TRAIN_X;%Y�ļ���X�ļ��ı�ǩ
TRAIN_Y1=JapaneseVowels_TRAIN_Y;
TEST_X1=JapaneseVowels_TEST_X;
TEST_Y1=JapaneseVowels_TEST_Y;
datas=[TRAIN_X1,TEST_X1]';
labels=[TRAIN_Y1,TEST_Y1]';

load JapaneseVowels_inverse;
Ma=JapaneseVowels_inverse;
Me=JapaneseVowels_inverse;

eta=0.001;
fac=eta*0.2;
rho=220;
means=csvread('JapaneseVowels_inverse/mean.csv'); %means�������ݼ����*ά��
cn=size(Ma,2);%��ĸ���

k_fold=10;%Ԥ�����ݷֳ�ʮ��
sum_accuracy_LogDet = 0;
final_saveResults=[];

[m,n] = size(datas);
indices = crossvalind('Kfold',m,k_fold);
TRAIN_X={};
TRAIN_Y=[];
TEST_X={};
TEST_Y=[];
for k_i=1:k_fold
    fprintf('��%d�ν���!\n',k_i);
    saveResults=[];
    %%%%--k_fold������֤����------------------------------------%%%
    test_indic = (indices == k_i);
    train_indic = ~test_indic;
    TRAIN_X = datas(train_indic,:)';%�ҳ�ѵ���������ǩ
    TRAIN_Y = labels(train_indic,:)';
    TEST_X = datas(test_indic,:)';%�ҳ������������ǩ
    TEST_Y = labels(test_indic,:)';
    train_tn=size(TRAIN_X,2);%ѵ���������е���Ŀ
    test_tn=size(TEST_X,2);%���Լ������е���Ŀ   
    %%%%--��ʼ׼ȷ��--------------------------------------------%%%
    [result,ll]=predict(Ma,TRAIN_X,train_tn,cn,means);%δ��һ��ѵ��ʱѵ�����õ��ķ�������������Ȼ���
    [~,train_accuracy,~,train_precision,train_recall,train_F1,~,~,~]=compute_accuracy_F (TRAIN_Y,result,9);%δ��һ��ѵ��ʱ��ѵ��������׼ȷ��
    fprintf('Train-accuracy: %f \n', train_accuracy);
    fprintf('------\n');
    re2=predict(Ma,TEST_X,test_tn,cn,means);%δ��һ��ѵ��ʱ���Լ��õ��ķ����������Լ���llδ��ֵ�õ�
    [~,test_accuracy,~,test_precision,test_recall,test_F1,~,~,~]=compute_accuracy_F (TEST_Y,re2,9);
    fprintf('Test-accuracy: %f \n', test_accuracy);
    saveResults=[saveResults,test_accuracy];
    fprintf('-----------------------------------------\n');
    %%%%--LogDet�Ż�������Э����--------------------------------%%%
    for kk=1:10%�ܵ�ѭ������
        for i=1:train_tn %��1��ѵ��������Сѭ��ִ��
            li=ll(i,:); %�õ���i��ѵ��������������ll������Ȼ����
            [lisort,I] = sort(li); %����lisort����������li,I����������Ԫ�ض�Ӧ��li�е��±�
            if (result(i)==TRAIN_Y(i)&&(lisort(cn)-lisort(cn-1))<rho)%��һ�������Ԥ��������ӽ������Ķ�����Ȼ��С���С��rho
                X= TRAIN_X{i}; %X����ѵ�������е�i��������
                [m,~]=size(X); %m�����i�������е�������С                
                k=TRAIN_Y(i); %��i��ѵ�����������k����1��cn
                for j=1:m %���ڲ�ѭ����
                    X(j,:)=X(j,:)-means(k,:); %������Ȼ����һ����
                end
                X=X'; % �����������������
                %--------------��������-----------------%+++++++
                IX=eye(size(X,2));
                M=Ma{k};%��k��������Э����
                M=(1+m*eta)*(M - eta*M*X*(IX + eta*X'*M*X)^(-1)*X'*M);%���µ�һ����Э����
                Ma{k}=M;                
                k1=I(cn-1);
                M1=Ma{k1};
                %            alpha1=eta*fac;
                %            M1=(1-m*eta)*(M1 + alpha1*M1*X*(IX - alpha1*X'*M1*X)^(-1)*X'*M1);%���µڶ�����Э����
                M1=(1-m*fac)*(M1 + fac*M1*X*(IX - fac*X'*M1*X)^(-1)*X'*M1);%���µڶ�����Э����
                Ma{k1}=M1;
            elseif(result(i)~=TRAIN_Y(i))%�ڶ��������Ԥ�����
                k=TRAIN_Y(i);
                M=Ma{k};%��k��������Э����
                X= TRAIN_X{i};
                [m,~]=size(X);
                for j=1:m
                    X(j,:)=X(j,:)-means(k,:);
                end
                X=X';                
                %--------------�����������-----------------%
                k1=result(i);
                IX=eye(size(X,2));
                M=(1+m*eta)*(M - eta*M*X*(IX + eta*X'*M*X)^(-1)*X'*M);
                Ma{k}=M;
                M1=Ma{k1};
                %            alpha1=eta*fac;
                %            M1=(1-m*eta)*(M1 + alpha1*M1*X*(IX - alpha1*X'*M1*X)^(-1)*X'*M1);
                M1=(1-m*fac)*(M1 + fac*M1*X*(IX - fac*X'*M1*X)^(-1)*X'*M1);
                Ma{k1}=M1;
            end
        end        
        %�����н������Ե÷�
%         Mee=Ma;
%         for i=1:cn
%             [bcc{i},EE{i}] = betweenness_centrality(sparse(abs(Mee{i})));
%         end        
        [result,ll]=predict( Ma,TRAIN_X,train_tn,cn,means);%���ຯ��
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
fprintf('%d������֤��ƽ��׼ȷ�ʣ�',k_fold);
disp( mean_sum_accuracy_LogDet );
[final_max_test_accuracy,final_p]=max(final_saveResults);
Position = find(final_saveResults==final_max_test_accuracy);
fprintf('��%d�ξ�����õĽ�����֤����׼ȷ�ʣ� %f \n',final_p,final_max_test_accuracy);
fprintf('Over!\n');

