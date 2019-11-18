function [ result,ll ] = predict(Ma,X,tn,cn,means)
%�����õ���������ǩresult�Ͷ�����Ȼll

ld=zeros(cn,1);%������Ȼll(Xi,Mai)��һ����
for i=1:cn
    ld(i,1)=log(det(Ma{i}));
end

result=zeros(tn,1);%�Է�����������ʼ��������Ϊ���ݼ���С
ll=zeros(tn,cn);%�Զ�����Ȼ�����ʼ������СΪ���ݼ���С*������
for i=1:tn  %��1�����ݼ���Сѭ��ִ��
    data_subsequencei=X{i};%��i������������
    [num_subsequencei_sample,~]=size(data_subsequencei);%��i�������е�������СThe number of subsequence sample
    for j=1:num_subsequencei_sample
        for k=1:cn  %kΪ��ǰ���
            x=data_subsequencei(j,:)-means(k,:);% (x-u)
            ll(i,k)=(-(x*Ma{k})*x'+ld(k,1))+ll(i,k);%��i�������ж�Ӧ��k����Ķ�����Ȼֵ
        end
    end
%     ll(i,:)=ll(i,:)/num_subsequencei_sample;
    [~,r]=max(ll(i,:));%����ɨ�裬�õ���i�������е�Ԥ�����rΪ��������Ȼֵ��Ӧ������±�
    result(i)=r;%�õ����������ݼ���Ԥ�����
end

end

