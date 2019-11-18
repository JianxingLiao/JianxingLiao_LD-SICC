clc
clear
close all
tic
% addpath('./LDMLT_TS')
addpath('./data')
addpath('./PrecisionRecall')
disp('Loading data...');

load JapaneseVowels_mergeData
load JapaneseVowels_TRAIN_X
load JapaneseVowels_TRAIN_Y
load JapaneseVowels_TEST_X
load JapaneseVowels_TEST_Y
% load JapaneseVowels_inverse;

mergeData=JapaneseVowels_mergeData;%ÿһ�����е������м���
TRAIN_X=JapaneseVowels_TRAIN_X;%TRAIN_X��ѵ��������
TRAIN_Y=JapaneseVowels_TRAIN_Y;%TRAIN_Y��ѵ������ǩ
TEST_X=JapaneseVowels_TEST_X;%TEST_X�ǲ��Լ�����
TEST_Y=JapaneseVowels_TEST_Y;%TEST_Y�ǲ��Լ���ǩ

%%---����GraphicalLasso�㷨�Ĳ���---%%
lambda=0.014;
maxIt=200;
tol=1e-4;

% %%---����LogDetɢ���Ż��㷨�Ĳ���---%%
% eta=0.001;
% fac=eta*0.2;
% rho=110;
%%---��þ�ֵ����-------------  ---------------%%
%%---��ͨ��GraphicalLasso�㷨�õ�ϡ����Э����--%%
classnumber=length(mergeData);%������
attributenumber=size(mergeData(1).data,2);%���Ը���
JapaneseVowels_means=[];%�����ֵ
JapaneseVowels_inverse=cell(1,classnumber);%����ϡ����Э����
for i=1:classnumber
    JapaneseVowels_means=[ JapaneseVowels_means;mean(mergeData(i).data) ];
end
save JapaneseVowels_means.mat JapaneseVowels_means
for i=1:classnumber
    S=cov(mergeData(i).data);
    [ JapaneseVowels_inverse{1,i},W{1,i} ] = graphicalLasso( S, lambda, maxIt, tol );    
end
save JapaneseVowels_inverse.mat JapaneseVowels_inverse
toc;
% %%---ͨ��LogDetɢ�Ƚ�һ���Ż�����ÿ������ϡ����Э����---%%
% Ma=JapaneseVowels_inverse;%������ÿ������ϡ����Э����
% Me=Ma;%������һ�ε�������Э�������
% saveresults=[];%�������յĽ��
% means=JapaneseVowels_means; %means�������ݼ����*ά��
% train_tn=size(TRAIN_X,2);%ѵ���������е���Ŀ
% test_tn=size(TEST_X,2);%���Լ������е���Ŀ
% cn=classnumber;%��ĸ���
% 
% [result,ll]=predict(Ma,TRAIN_X,train_tn,cn,means);%δ��һ��ѵ��ʱѵ�����õ��ķ�������������Ȼ���
% [~,train_accuracy,~,train_precision,train_recall,train_F1,~,~,~]=compute_accuracy_F (TRAIN_Y,result,cn);%δ��һ��ѵ��ʱ��ѵ��������׼ȷ��
% fprintf('Train-accuracy: %f \n', train_accuracy);
% fprintf('------\n');
% re2=predict(Ma,TEST_X,test_tn,cn,means);%δ��һ��ѵ��ʱ���Լ��õ��ķ����������Լ���llδ��ֵ�õ�
% [~,test_accuracy,~,test_precision,test_recall,test_F1,~,~,~]=compute_accuracy_F (TEST_Y,re2,cn);
% fprintf('Test-accuracy: %f \n', test_accuracy);
% fprintf('-----------------------------------------\n');
% test_accuracy_old=test_accuracy;%������һ������׼ȷ��
% 
% %----------ѵ������--------------%
% for kk=1:30%�ܵ�ѭ������
%     for i=1:train_tn %��1��ѵ��������Сѭ��ִ��
%         li=ll(i,:); %�õ���i��ѵ��������������ll������Ȼ����
%         [lisort,I] = sort(li); %����lisort����������li,I����������Ԫ�ض�Ӧ��li�е��±�
%         if (result(i)==TRAIN_Y(i)&&(lisort(cn)-lisort(cn-1))<rho)%��һ�������Ԥ��������ӽ������Ķ�����Ȼ��С���С��rho
%             X= TRAIN_X{i}; %X����ѵ�������е�i��������
%             [m,~]=size(X); %m�����i�������е�������С
%             k=TRAIN_Y(i); %��i��ѵ�����������k����1��cn
%             X1=zeros(size(X));
%             for j=1:m %���ڲ�ѭ����
%                 X1(j,:)=X(j,:)-means(k,:); %������Ȼ����һ����
%             end
%             X1=X1'; % �����������������
%             %--------------��������-----------------%
%             IX=eye(size(X1,2));
%             M=Ma{k};%��k��������Э����
%             M=(1+m*eta)*(M - eta*M*X1*(IX + eta*X1'*M*X1)^(-1)*X1'*M);%���µ�һ����Э����
%             log_likelihood_old=ll(i,k);
%             log_likelihood_new=log_likelihood(TRAIN_X{i},M,means(k,:));
%             if(log_likelihood_old<log_likelihood_new)
%                 Ma{k}=M;
%             end
% %             fprintf('��һ�������%d_%d log_likelihood_old:%f \n',kk,i,log_likelihood_old);
% %             fprintf('��һ�������%d_%d log_likelihood_new:%f \n',kk,i,log_likelihood_new);
%             k1=I(cn-1);
%             X2=zeros(size(X));
%             for j=1:m
%                 X2(j,:)=X(j,:)-means(k1,:);
%             end
%             X2=X2';
%             M1=Ma{k1};
%             M1=(1-m*fac)*(M1 + fac*M1*X2*(IX - fac*X2'*M1*X2)^(-1)*X2'*M1);%���µڶ�����Э����
%             log_likelihood_old=ll(i,k1);
%             log_likelihood_new=log_likelihood(TRAIN_X{i},M1,means(k1,:));
%             if(log_likelihood_old>log_likelihood_new)
%                 Ma{k1}=M1;
%             end
% %             fprintf('�ڶ��������%d_%d log_likelihood_old:%f \n',kk,i,log_likelihood_old);
% %             fprintf('�ڶ��������%d_%d log_likelihood_new:%f \n',kk,i,log_likelihood_new);
%         elseif(result(i)~=TRAIN_Y(i))%�ڶ��������Ԥ�����
%             k=TRAIN_Y(i);%��ȷ�ı�ǩ
%             M=Ma{k};
%             X= TRAIN_X{i};
%             [m,~]=size(X);
%             X1=zeros(size(X));
%             for j=1:m
%                 X1(j,:)=X(j,:)-means(k,:);
%             end
%             X1=X1';
%             
%             %--------------�����������-----------------%
%             
%             IX=eye(size(X1,2));
%             M=(1+m*eta)*(M - eta*M*X1*(IX + eta*X1'*M*X1)^(-1)*X1'*M);
%             log_likelihood_old=ll(i,k);
%             log_likelihood_new=log_likelihood(TRAIN_X{i},M,means(k,:));
%             if(log_likelihood_old<log_likelihood_new)
%                 Ma{k}=M;
%             end
% %             fprintf('��һ�������%d_%d log_likelihood_old:%f \n',kk,i,log_likelihood_old);
% %             fprintf('��һ�������%d_%d log_likelihood_new:%f \n',kk,i,log_likelihood_new);
%             k1=result(i);%��������ǩ
%             X2=zeros(size(X));
%             for j=1:m
%                 X2(j,:)=X(j,:)-means(k1,:);
%             end
%             X2=X2';
%             M1=Ma{k1};
%             M1=(1-m*fac)*(M1 + fac*M1*X2*(IX - fac*X2'*M1*X2)^(-1)*X2'*M1);
%             log_likelihood_old=ll(i,k1);
%             log_likelihood_new=log_likelihood(TRAIN_X{i},M1,means(k1,:));
%             if(log_likelihood_old>log_likelihood_new)
%                 Ma{k1}=M1;
%             end
% %             fprintf('�ڶ��������%d_%d log_likelihood_old:%f \n',kk,i,log_likelihood_old);
% %             fprintf('�ڶ��������%d_%d log_likelihood_new:%f \n',kk,i,log_likelihood_new);
%         end
%     end
%     
%     [result,ll]=predict( Ma,TRAIN_X,train_tn,cn,means);%���ຯ��
%     [~,train_accuracy,~,train_precision,train_recall,train_F1,~,~,~]=compute_accuracy_F (TRAIN_Y,result,cn);
%     fprintf('Train-accuracy: %f \n', train_accuracy);
%     fprintf('------%d\n',kk);
%     re2=predict( Ma,TEST_X,test_tn,cn,means);
%     [~,test_accuracy,~,test_precision,test_recall,test_F1,~,~,~]=compute_accuracy_F (TEST_Y,re2,cn);
%     fprintf('Test-accuracy: %f \n', test_accuracy);
%     fprintf('------------------------\n');
%     saveresults=[saveresults,test_accuracy];
% end
% [max_test_accuracy,p]=max(saveresults);
% Position = find(saveresults==max_test_accuracy);
% fprintf('����÷���׼ȷ�ʵ�ѭ��������');
% disp(Position);
% fprintf('max_test_accuracy: %f \n', max_test_accuracy);
% toc;