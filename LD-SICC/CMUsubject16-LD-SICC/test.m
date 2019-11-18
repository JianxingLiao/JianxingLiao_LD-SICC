clc;
clear;
addpath('./data')
load CMUsubject16_TRAIN_X
load CMUsubject16_TRAIN_Y
load CMUsubject16_TEST_X
load CMUsubject16_TEST_Y
load CMUsubject16_inverse;
load CMUsubject16_means;

TRAIN_X=CMUsubject16_TRAIN_X;%TRAIN_X��ѵ��������
TRAIN_Y=CMUsubject16_TRAIN_Y;%TRAIN_Y��ѵ������ǩ
TEST_X=CMUsubject16_TEST_X;%TEST_X�ǲ��Լ�����
TEST_Y=CMUsubject16_TEST_Y;%TEST_Y�ǲ��Լ���ǩ

eta=0.001;
fac=eta*0.2;
rho=100;

Ma=CMUsubject16_inverse;
Me=CMUsubject16_inverse;


    saveResults=[];%�������յĽ��
    means=CMUsubject16_means; %means�������ݼ����*ά��
    train_tn=size(TRAIN_X,2);%ѵ���������е���Ŀ
    test_tn=size(TEST_X,2);%���Լ������е���Ŀ
    cn=size(Ma,2);%��ĸ���
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
    for kk=1:20%�ܵ�ѭ������
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
                %             fprintf('��һ�������%d_%d log_likelihood_old:%f \n',kk,i,log_likelihood_old);
                %             fprintf('��һ�������%d_%d log_likelihood_new:%f \n',kk,i,log_likelihood_new);
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
        re2=predict( Ma,TEST_X,test_tn,cn,means);
        [~,test_accuracy,~,test_precision,test_recall,test_F1,~,~,~]=compute_accuracy_F (TEST_Y,re2,cn);
        fprintf('Test-accuracy: %f \n', test_accuracy);
        saveResults=[saveResults,test_accuracy];
        fprintf('------------------------\n');
    end

[max_test_accuracy,p]=max(saveResults);
Position = find(saveResults==max_test_accuracy);
fprintf('����÷���׼ȷ�ʵ�ѭ��������');
disp(Position);
fprintf('max_test_accuracy: %f \n', max_test_accuracy);
fprintf('Over!\n');

