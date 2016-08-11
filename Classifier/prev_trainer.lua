require 'Csv'
require 'nn'

local csvData  = Csv('line_data.csv','r')

local data = csvData:readall()
tensorData = torch.DoubleTensor(data)

mlp = nn.Sequential();
local feature = tensorData[{ {},{1,2}}]


dataset ={};
function dataset:size() return tensorData:size(1) end

for i=1,tensorData:size(1) do
	local input =torch.DoubleTensor(1,1):fill(tensorData[i][1]/100)
	local output =torch.DoubleTensor(1,1):fill(tensorData[i][2]/200)
	dataset[i] = {input, output};
	
end

inputs=1;
outputs=1;
mlp:add(nn.Linear(inputs,outputs))

criterion = nn.MSECriterion()
print("1")

trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate =0.01
trainer.maxIteration = 300
print("2")
trainer:train(dataset);
