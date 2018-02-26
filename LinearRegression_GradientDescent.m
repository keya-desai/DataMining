filename='boston_housing.csv';
M=csvread(filename,1,0,[1 0 506 12]);
y=csvread(filename,1,13);
N=M;
range=max(M)-min(M);
temp=mean(M);
for b=1:506
    for c=1:13
        newx = (M(b,c)-temp(1,c))/range(1,c);
        M(b,c)=newx;
     end
end
theta=ones(14,1);
oldcost=0;
it=0;
M=[ones(506,1) M];
for k=1:300
    sum=zeros(1,14);
    for p=1:506
        h=(M(p,:))*theta;
        o=h(1,1)-y(p,1);
        for t=1:14
            for w=1:506
                op=o*M(w,t);
                sum(1,t)=sum(1,t)+op;
            end
        end
    end
    dif=sum/506;
    alpha=0.03;
    dif = alpha*dif;
    newtheta=theta-alpha*transpose(dif);
    s=0;
for p=1:506
    h=(M(p,:))*newtheta;
    square=(h(1,1)-y(t,1)).^2;
    s=s+square;
end
cost=s/1012;
theta=newtheta;
    if(k==0)
        oldcost=cost;
        continue;
    else
       if(cost-oldcost<0.00001 && cost-oldcost>0)
           it=it+1;
           disp('alpha is :');
           disp(alpha);
           disp('Converging value of cost function is :');
           disp(cost);
           break;
       else
           it=it+1;
           oldcost=cost;
       end
    end
end
disp('Iterations:');
disp(it);
if(it==300)
    disp('alpha is :');
    disp(alpha);
    disp('Cost function diverges');
end
 
a1= [0.0101, 30, 5.19, 0, 0.0493, 6.059,37.3, 4.8122,1, 430,19.6, 375.21, 8.51 ] ;
a2=[ 0.02501, 35, 4.15, 1, 0.77, 8.78, 81.3, 2.5051, 24, 666, 17, 382.8, 11.48 ];
a3=[ 3.67822, 0, 18.1, 1, 0.7, 6.649, 98.8, 1.1742, 24, 711, 20.2, 398.28, 18.07 ];
%range1=max(a1)-min(a1);
    for c=1:13
        newx1 = (a1(1,c)-temp(1,c))/range(1,c);
        newx2 = (a2(1,c)-temp(1,c))/range(1,c);
        newx3 = (a3(1,c)-temp(1,c))/range(1,c);
        a1(1,c)=newx1;
        a2(1,c)=newx2;
        a3(1,c)=newx3;
    end
    a1=[ones(1,1) a1];
    a2=[ones(1,1) a2];
    a3=[ones(1,1) a3];
    h1=a1*theta;
    h2=a2*theta;
    h3=a3*theta;
    disp('predicted price of 1st example: ');
    disp(h1);
    disp(' predicted price of 2nd example: ');
    disp(h2);
    disp(' predicted price of 3rd example: ');
    disp(h3);
