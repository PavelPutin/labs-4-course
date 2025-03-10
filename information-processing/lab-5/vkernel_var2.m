function p_=vkernel_var2(x,XN,h_N,kl_kernel)
%������� ��� ����������� ���� � ��������� ������ 
%��������� ������������� ������������ ������� �������
%x-������ �������� (�����), ��� ������� ������������ ���������
%XN-������� ��������� ������� ������
%h_N- ��������, ������������ ������ ������� ����������� ���� (������� �������)
%kl_kernel-���� ��� ����������� ���� ������������� ����
%kl_kernel=11 - ����������� ������� c �������������� ������������ �������;
%kl_kernel=12 - ����������� ������� c �������������� ������� ����������;
%kl_kernel=2 - ������������� �������;    
%kl_kernel=3 - ������� ������������� �������;
%kl_kernel=4 - ������� ����������� �������;
% ����� �����
%kl_kernel=5 - ������� ������� (���.2.�)
%kl_kernel=6 - ������� ������� (���.2.�)

[n1,mx]=size(x);%����������� ������������ � ����� �����, ��� ������� ���������� ������
[n2,N]=size(XN);%����������� ������������ � ����� ��������� ������� 
if n1==n2, n=n1;
    p_=zeros(1,mx); C=eye(n,n); C_=C;
    if kl_kernel==12 && N>1, %������ ���������� ������� ���������� 
        C=zeros(n,n); m_=mean(XN')';
        for i=1:N,
             C=C+(XN(:,i)-m_)*(XN(:,i)-m_)';
        end;
        C=C/(N-1); C_=C^-1;
    end;
    %���������� �������� ������� ���� � �������� XN(:,1:N) � ������ x(:,1:mx)
    %��� N=1 �������� ��������� ������� ���� c � ������� XN(:,1)
    fit=zeros(N,mx); 
    for i=1:N,
        p_k=zeros(n,mx); mx_i=repmat(XN(:,i),[1,mx]);
        switch kl_kernel,
            case 11,
                ro=sum((x-mx_i).^2,1); 
                fit(i,:)=exp(-ro/(2*h_N^2))/((2*pi)^(n/2)*(h_N^n));
            case 12,
                ro=sum((C_*(x-mx_i)).*(x-mx_i),1);
                fit(i,:)=exp(-ro/(2*h_N^2))*((2*pi)^(-n/2)*(h_N^-n)*(det(C)^-0.5));
            case 2,
                ro=abs(x-mx_i)/h_N; 
                fit(i,:)=prod(exp(-ro),1)/(2*h_N^n); 
            case 3,
                ro=abs(x-mx_i)/h_N; 
                for k=1:n, 
                    ind=logical(ro(k,:)<1); p_k(k,ind)=1/2;
                end;
                fit(i,:)=prod(p_k,1)/h_N^n;
            case 4,
                ro=abs(x-mx_i)/h_N; 
                for k=1:n, 
                   ind=logical(ro(k,:)<1); p_k(k,ind)=(1-ro(k,ind));
                end;
                fit(i,:)=prod(p_k,1)/h_N^n;
            % ����� ����������� ����� �����
            case 5, % ���. 2.�)
                ro=(x-mx_i)/h_N;
                fit(i,:)=1/(pi*h_N)*1./(1+ro.^2);   % ������� ����� �������� � ����������� � �������!
            case 6, % ���. 2.�)
                ro=(x-mx_i)/(2*h_N); 
                fit(i,:)=1/(2*pi*h_N)*(sin(ro)./ro).^2; 
        end; %kl_kernel
    end;%i=1:N
    %���������� ������ ��������� ������������� ������������
    if N>1, p_=sum(fit)/N; else p_=fit; end;
else %n1=~n2
    error('����������� ������ (n1 � n2) �� ���������');
end;
