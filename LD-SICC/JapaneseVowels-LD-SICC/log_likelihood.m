function log_likelihood = log_likelihood(X,Ma_label,means_label)
%�����õ�������Ȼll
%Ma ��Э�������
%means ��ֵ����
%X_label �����е����
%X ������
%X_num �������ڲ��Լ��ı��
%cn ��ĸ���

ld=log(det(Ma_label));%������Ȼll(Xi,Mai)��һ����logdet(Ma{})
ll=0;

[num_subsequencei_sample,~]=size(X);%������X��������СThe number of subsequence sample
for j=1:num_subsequencei_sample
    x=X(j,:)-means_label;% (x-u)
    ll=-(x*Ma_label)*x'+ld+ll;%��i�������ж�Ӧ��k����Ķ�����Ȼֵ
end
log_likelihood=ll;%����ɨ�裬�õ���i�������е�Ԥ�����rΪ��������Ȼֵ��Ӧ������±�
end