require 'Csv'
require 'nn'

local csvData  = Csv('line_data.csv','r')

local data = csvData:readall()
tensorData = torch.DoubleTensor(data)
--[[print(tensorData) --]]
mlp = nn.Sequential();
local feature = tensorData[{ {},{1,2}}]
--[[(feature)  --]]

dataset ={};
function dataset:size() return tensorData:size(1) end
--[[print(tensorData:size(1))--]]
for i=1,tensorData:size(1) do
	local input =torch.DoubleTensor(1,1):fill(tensorData[i][1]/100)
	local output =torch.DoubleTensor(1,1):fill(tensorData[i][2]/200)
	dataset[i] = {input, output};
	
end
--[[
dataset ={};
function dataset:size() return 100 end
for i =1,dataset:size() do
	local input = torch.randn(2);
	local output = torch.Tensor(1)
	if input[1]*input[2] >0 then
		output[1] =-1;
	else
		output[1] =1;
	end
	dataset[i] = {input, output};
end
--]]
--[[print(dataset)--]]

inputs=1;
outputs=1;
mlp:add(nn.Linear(inputs,outputs))

criterion = nn.MSECriterion()
print("1")

trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate =0.01
trainer.maxIteration = 300
print("2")
--[[print(dataset)
print(dataset[21][2])--]]
trainer:train(dataset);
