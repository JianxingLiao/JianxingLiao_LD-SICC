function Ma = LDMLT_TS(X,Y, params)
% ---------------------------------------------------------------------------------------
% SIGNATURE
% ---------------------------------------------------------------------------------------
% Author: Jiangyuan Mei
% E-Mail: meijiangyuan@gmail.com
% Date  : Sep 23 2014
% ---------------------------------------------------------------------------------------

%generate Mahalanobis Distance  M which based Dynamic Time Warping Measure
%using LogDet Divergence based Metric Learning with Triplet Constraints
%algorithms
%Input:     X             --> X{i} (i=1,2,...,n) is an D x t matrix
%           Y             --> label
%           params        --> structure containing the parameters
%           params.tripletsfactor;  --> (quantity of triplets in each cycle) = params.tripletsfactor x (quantity of training instances)
%           params.cycle;           --> the maximum cycle of metric learning process
%           params.alphafactor;     --> alpha = params.alphafactor/(quantity of triplets in each cycle)
%Output:    M             --> learned PSD matrix

% Jiangyuan Mei, Meizhu Liu, Hamid Reza Karimi, and Huijun Gao, 
%"LogDet Divergence based Metric Learning with Triplet Constraints 
% and Its Applications", IEEE Transactions on image processing, Accepted.

% Jiangyuan Mei, Meizhu Liu, Yuan-Fang Wang, and Huijun Gao, 
%"Learning a Mahalanobis Distance based Dynamic
%Time Warping Measure for Multivariate Time Series
%Classification". 



if (~exist('params')),
    params = struct();
    params = SetDefaultParams(params);
end

numberCandidate =size(X,2);
numberFeature=size(X{1},2);

%The Mahalanobis matrix M starts from identity matrix
% M=eye(numberFeature,numberFeature);
% Ma={};
% lable=1;
% s=[];
% for i=1:size(X,2)
%     if lable ~= Y(i)
%         Me=cov(s);
%         Me=inv(Me);
%         Ma=[Ma Me];
%         lable=lable+1;
%         s=[];
%         s=[s;X{i}];
%     else
%         s=[s;X{i}];
%     end
% end
% Me=cov(s);
% Me=inv(Me);
% Ma=[Ma Me]; 
load inverse;
Ma=inverse;
load JapaneseVowels_TEST_X.mat;
TEST_X=JapaneseVowels_TEST_X;
load JapaneseVowels_TEST_Y
TEST_Y=JapaneseVowels_TEST_Y;

% Get all the lables of the data
Y_kind=unique(Y);
[X,Y]=data_rank(X,Y,Y_kind); % rank the original data according to their label 根据标签排序

%S record whether dissimilar or not
S = zeros(numberCandidate,numberCandidate); 
for i=1:numberCandidate
    for j=1:numberCandidate
        if Y(i)==Y(j)
            S(i,j) = 1;
        end
    end
end



[Triplet,rho,Error_old]=Select_Triplets(X,params.tripletsfactor,Ma,Y,S); % dynamic triplets building strategy
iter=size(Triplet,1);    
total_iter=iter;

result=predict(Ma,TEST_X);
[~,accuracy,~,~,~,~,~,~,~]=compute_accuracy_F (TEST_Y,result,9);
fprintf('accuracy: %f \n',accuracy);
for i=1:params.cycle
    alpha=params.alphafactor/iter;%随着迭代次数增加，学习率越来越低
    rho=0;
    Ma=update_M(Ma,X,Triplet,alpha,rho);  % update the Mahalanobis matrix M
    result=predict(Ma,TEST_X);
    [~,accuracy,~,~,~,~,~,~,~]=compute_accuracy_F (TEST_Y,result,9);
    fprintf('accuracy: %f \n',accuracy);
    [Triplet,~,Error_new]=Select_Triplets(X,params.tripletsfactor,Ma,Y,S); % dynamic triplets building strategy
%     iter=size(Triplet,1);
%     total_iter=total_iter+iter; % record the toatl ietrations in metric learning process.
%     params.tripletsfactor=Error_new/Error_old*params.tripletsfactor; % the quantity of triplets reduces with the shrink of error
%     co=(Error_old-Error_new)/Error_old;
%     if abs(co)<10e-5
%         break;
%     end
%     fprintf('Cycle: %d, Error: %d, tol: %f,iteration: %d\n', i, Error_new, co,iter);
%     Error_old=Error_new;
end
fprintf('LDMLT converged to error: %d, total cycle: %d, total iteration: %d \n', Error_new,i,total_iter);


% The proposed LDMLT algorithm
function Ma=update_M(Ma,X,triplet,gamma,rho)
% M=cov(X{triplet(1,1)});
% M=M^(-1);
% M=M/trace(M);%为什么需要？
i=1;
options=zeros(1,5);
options(5) = 1;
while (i<size(triplet,1))
%     fprintf('这是第%d次循环 \n',i);
    i1 = triplet(i,1);
    i2 = triplet(i,2);
    i3 = triplet(i,3);
    X1=(X{i1});
    X2=(X{i2});
    X3=(X{i3});
    cls=floor((i1-1)/30)+1;
    M=Ma{cls};
    means=csvread('inverse/mean.csv');
    meanx1=means(cls,:);
%     meanx1=mean(X1,1);
%     meanx2=mean(X2,1);
%     meanx3=mean(X3,1);
    for k=1:size(X1,1)
        X1(k,:)=X1(k,:)-meanx1;
    end
    for k=1:size(X2,1)
        X2(k,:)=X2(k,:)-meanx1;
    end
    for k=1:size(X3,1)
        X3(k,:)=X3(k,:)-meanx1;
    end
    Dist1=0.5*abs(trace(X2*M*X2')/size(X2,1)-trace(X1*M*X1')/size(X1,1));
    Dist2=0.5*abs(trace(X3*M*X3')/size(X3,1)-trace(X1*M*X1')/size(X1,1));
    X1=X1';
    X2=X2';
    X3=X3';
%     [Dist1,swi1,swi2]=dtw_metric(X{i1},X{i2},M); %swi1是经过对整齐后的时间序列段，维度是d*p 
%     %P=swi1-swi2;
%     [Dist2,swi3,swi4]=dtw_metric(X{i1},X{i3},M);
%     %Q=swi1-swi3;
%     Px1=zeros(size(swi1));
%     Py=zeros(size(swi2));
%     Px2=zeros(size(swi3));
%     Pz=zeros(size(swi4));
%     for j=1:size(swi1,2)
%         Px1(:,j)=swi1(:,j)-mean(X1,2);
%     end
%     for j=1:size(swi2,2)
%         Py(:,j)=swi2(:,j)-mean(X2,2);
%     end
%     for j=1:size(swi3,2)
%         Px2(:,j)=swi3(:,j)-mean(X1,2);
%     end
%     for j=1:size(swi4,2)
%         Pz(:,j)=swi4(:,j)-mean(X3,2);
%     end
    IX1=eye(size(X1,2));
    IX2=eye(size(X2,2));
    IX3=eye(size(X3,2));
    fprintf('Dist2-Dist1: %f, rho: %d \n', Dist2-Dist1,rho);
    if Dist2-Dist1<0 
        if (trace(X2'*M*X2)/size(X2,2)>=trace(X1'*M*X1)/size(X1,2))&&(trace(X3'*M*X3)/size(X3,2)>=trace(X1'*M*X1)/size(X1,2))
            setlmis([]);
            alpha = lmivar(1,[1 1]);
            lmiterm([-1 1 1 alpha],1,0.5/size(X2,2)*(X2*X2')-0.5/size(X3,2)*(X3*X3'));
            lmiterm([-1 1 1 0],M^-1);
            lmiterm([-2 1 1 alpha],1,1);
            lmiterm([2 1 1 0],0);
            lmis = getlmis;
            [~,xfeas] = feasp(lmis,options);
            if size(xfeas,1)==0
                alpha=0;
            else
                alpha=gamma*xfeas;
            end
            %alpha=gamma/trace((eye(size(M,1))-M)^(-1)*M*Q*Q');%这句代码什么意思？
            M_temp=M - 0.5*alpha/size(X2,2)*M*X2*(IX2 + 0.5*alpha/size(X2,2)*X2'*M*X2)^(-1)*X2'*M;
            M = M_temp + 0.5*alpha/size(X3,2)*M_temp*X3*size(X3,2)*(IX3 - 0.5*alpha/size(X3,2)*X3'*M_temp*X3)^(-1)*X3'*M_temp;
%             [L S R]=svd(M);
%             M=M/sum(trace(S));
%             M=M/trace(M);
        elseif (trace(X2'*M*X2)/size(X2,2)>=trace(X1'*M*X1)/size(X1,2))&&(trace(X3'*M*X3)/size(X3,2)<trace(X1'*M*X1)/size(X1,2))
            setlmis([]);
            alpha = lmivar(1,[1 1]);
            lmiterm([-1 1 1 alpha],1,0.5/size(X2,2)*(X2*X2')-0.5/size(X3,2)*(X3*X3')-(X1*X1')/size(X1,2));
            lmiterm([-1 1 1 0],M^-1);
            lmiterm([-2 1 1 alpha],1,1);
            lmiterm([2 1 1 0],0);
            lmis = getlmis;
            [tmin,xfeas] = feasp(lmis,options);
            if size(xfeas,1)==0
                alpha=0;
            else
                alpha=gamma*xfeas;
            end
            %alpha=gamma/trace((eye(size(M,1))-M)^(-1)*M*Q*Q');%这句代码什么意思？
            M_temp1=M - 0.5*alpha/size(X2,2)*M*X2*(IX2 + 0.5*alpha/size(X2,2)*X2'*M*X2)^(-1)*X2'*M;
            M_temp2=M_temp1 + alpha/size(X1,2)*M_temp1*X1*(IX1 - alpha/size(X1,2)*X1'*M_temp1*X1)^(-1)*X1'*M_temp1;
            M = M_temp2 + 0.5*alpha/size(X3,2)*M_temp2*X3*(IX3 - 0.5*alpha/size(X3,2)*X3'*M_temp2*X3)^(-1)*X3'*M_temp2;
%             [L S R]=svd(M);
%             M=M/sum(trace(S));
%             M=M/trace(M);
        elseif (trace(X2'*M*X2)/size(X2,2)<trace(X1'*M*X1)/size(X1,2))&&(trace(X3'*M*X3)/size(X3,2)>=trace(X1'*M*X1)/size(X1,2))
            setlmis([]);
            alpha = lmivar(1,[1 1]);
            lmiterm([-1 1 1 alpha],1,-0.5/size(X2,2)*(X2*X2')-0.5/size(X3,2)*(X3*X3')+(X1*X1')/size(X1,2));
            lmiterm([-1 1 1 0],M^-1);
            lmiterm([-2 1 1 alpha],1,1);
            lmiterm([2 1 1 0],0);
            lmis = getlmis;
            [tmin,xfeas] = feasp(lmis,options);
            if size(xfeas,1)==0
                alpha=0;
            else
                alpha=gamma*xfeas;
            end
            %alpha=gamma/trace((eye(size(M,1))-M)^(-1)*M*Q*Q');%这句代码什么意思？
            M_temp1=M - alpha/size(X1,2)*M*X1*(IX1 + alpha/size(X1,2)*X1'*M*X1)^(-1)*X1'*M;
            M_temp2=M_temp1 + 0.5*alpha/size(X2,2)*M_temp1*X2*(IX2 - 0.5*alpha/size(X2,2)*X2'*M_temp1*X2)^(-1)*X2'*M_temp1;
            M = M_temp2 + 0.5*alpha/size(X3,2)*M_temp2*X3*(IX3 - 0.5*alpha/size(X3,2)*X3'*M_temp2*X3)^(-1)*X3'*M_temp2;
%             [L S R]=svd(M);
%             M=M/sum(trace(S));
%             M=M/trace(M);
        elseif (trace(X2'*M*X2)/size(X2,2)<trace(X1'*M*X1)/size(X1,2))&&(trace(X3'*M*X3)/size(X3,2)<trace(X1'*M*X1)/size(X1,2))
            setlmis([]);
            alpha = lmivar(1,[1 1]);
            lmiterm([-1 1 1 alpha],1,-0.5/size(X2,2)*(X2*X2')+0.5/size(X3,2)*(X3*X3'));
            lmiterm([-1 1 1 0],M^-1);
            lmiterm([-2 1 1 alpha],1,1);
            lmiterm([2 1 1 0],0);
            lmis = getlmis;
            [tmin,xfeas] = feasp(lmis,options);
            if size(xfeas,1)==0
                alpha=0;
            else
                alpha=gamma*xfeas;
            end
            %alpha=gamma/trace((eye(size(M,1))-M)^(-1)*M*Q*Q');%这句代码什么意思？
            M_temp=M - 0.5*alpha/size(X3,2)*M*X3*(IX3 + 0.5*alpha/size(X3,2)*X3'*M*X3)^(-1)*X3'*M;
            M = M_temp + 0.5*alpha/size(X2,2)*M_temp*X2*(IX2 - 0.5*alpha/size(X2,2)*X2'*M_temp*X2)^(-1)*X2'*M_temp;
%             [L S R]=svd(M);
%             M=M/sum(trace(S));
%             M=M/trace(M);
        end
    end   
    i = i + 1;
    Ma{cls}=M;
end
% M=M*size(M,1);

function [X,Y]=data_rank(X,Y,Y_kind)

X_data=[];
Y_data=[];
for l=1:length(Y_kind)
    index=find(Y==Y_kind(l)); %找到所有当前类别的样本
    X_data=[X_data X(index)]; %矩阵合并，合并不同类别的样本，将同一个类别的样本放在一起
    Y_data=[Y_data Y(index)];
end
X=X_data;
Y=Y_data;
