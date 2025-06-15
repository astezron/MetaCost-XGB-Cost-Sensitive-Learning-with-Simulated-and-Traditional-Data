[filetitle,nfield,header,A] = importfile('Analysis1.csv',-1,1,1:2);    % Proxy results: first column = 0 (false) or 1 (true); second column = frequency of true across the realizations
[filetitle,nfield,header,B] = importfile('Analysis2.csv',-1,1,1:2);    % Bootstrap results: first column = 0 (false) or 1 (true); second column = frequency of true across the realizations

% Default parameter
n = 1000;   % number of closest data for moving window statistics

%--------------------------------------

k = 1;
for p = 0.4:0.01:1
    [ignore,I] = sort(abs(A(:,2)-p));
    index = I(1:n);
    t1(k,:) = mean(A(index,:));
    [ignore,I] = sort(abs(B(:,2)-p));
    index = I(1:n);
    t2(k,:) = mean(B(index,:));
    k = k+1;
end
figure(1);
clf;
set(gcf,'DefaultAxesFontName','Times','DefaultAxesFontSize',14);
hold on;
plot([0.4 1],[0.4,1],'k-','LineWidth',2);
plot(t1(:,2),t1(:,1),'b-','LineWidth',2);
plot(t2(:,2),t2(:,1),'r-','LineWidth',2);
axis([0.4 1 0.4 1]);
grid
xlabel('Correct classification probability')
ylabel('Correct classification frequency')
set(gca,'FontName','Times','FontSize',14)
legend('Identity','Test dataset with simulated proxies','Test dataset with bootstrapped classifiers','Location','northwest')
print('-dpng','Analysis','-r300');
