function[]=marleneaxes()
set(gca,'tickdir','out')
a=get(gca,'ticklength');
set(gca,'ticklength',[a(1)*2,a(2)*2])
box off
