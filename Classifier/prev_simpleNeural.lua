require 'Csv'
require 'nn'

csvData  = Csv('line_data.csv','r')

data = csvData:readall()
tensorData = torch.DoubleTensor(data);
input = tensorData:select(2,1);
inputReal = torch.DoubleTensor(input:size(1),1);
for i =1, input:size(1) do
	inputReal[i][1]= input[i]/100
end
target = tensorData:select(2,2);
targetReal = torch.DoubleTensor(target:size(1),1);
for i =1, target:size(1) do
	targetReal[i][1]= target[i]/200;
end

mlp = nn.Sequential();

inputs=1;
outputs=1;
mlp:add(nn.Linear(inputs,outputs))
print(inputReal)
print(target)
criterion = nn.MSECriterion()


for i = 1,3000 do
	mlp:updateParameters(0.1)
	output = mlp:forward(inputReal);
	loss = criterion:forward(output, targetReal);
	print(loss);
	mlp:zeroGradParameters();
	grad_output = criterion:backward(output, targetReal)
	mlp:backward(inputReal,grad_output)

end


