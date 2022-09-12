
clear all;
d = [0.3 0.5 0.8];
et= [1:4];    % 1--N(0,1);  2--Cauchy(0,1);  3--exp(1);  4--t(2)
l = length(d); 
L = length(et);
for i=1:l 
    for j=1:L
      if (j==1)
           Name = ['Normal',num2str(d(i)),'.txt'];
      end
        
        if (j==2)
           Name = ['Cauchy',num2str(d(i)),'.txt'];
        end
       
        if (j==3)
           Name = ['Exp',num2str(d(i)),'.txt'];
        end
       
        if (j==4)
           Name = ['T',num2str(d(i)),'.txt'];
        end

      Result = Runcode(d(i),et(j));
      %dlmwrite(Name, Result, 'precision', '%.8f','delimiter', '\t','newline', 'pc')
      save(Name,'Result','-ascii');
    end
end
