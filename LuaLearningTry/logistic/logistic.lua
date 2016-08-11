require 'cunn'
--require 'Csv'
require 'optim'

require 'csvigo'

print("\n\n\n------------------------Start------------------------")
loaded = csvigo.load('original.csv');

brands = torch.DoubleTensor(loaded.brand)
females = torch.DoubleTensor(loaded.female)
ages = torch.DoubleTensor(loaded.age)

ages = (ages - torch.min(ages))/ (torch.max(ages) - torch.min(ages))


dataset_inputs =  torch.DoubleTensor((#brands)[1],2)
dataset_inputs[{{},1}] = females
dataset_inputs[{{},2}] = ages

dataset_outputs = brands
print("------------------------Data Collected------------------------\n\n")
print("------------------------Creating model------------------------\n\n")

linlayer = nn.Linear(2,3)
softMaxLayer = nn.LogSoftMax()

model = nn.Sequential()
model:add(linlayer)
model:add(softMaxLayer)
criterion = nn.ClassNLLCriterion()

x, dl_dx = model:getParameters()
print("------------------------Model Created------------------------\n\n")
print("------------------------Creating evalFunction------------------------\n\n")

feval = function(x_new)
   if x ~= x_new then
	x:copy(x_new)
   end
   _nidx_ = (_nidx_ or 0) + 1
   
   if _nidx_ > ((#dataset_inputs)[1]) then _nidx_ = 1 end
	
   local inputs = dataset_inputs[_nidx_]
   local target = dataset_outputs[_nidx_]
   dl_dx:zero()				--Gradient Zero

print (inputs:size())
print (target)

   local loss_x = criterion:forward(model:forward(inputs),target)
   model:backward(inputs, criterion:backward(model.output, target))
	return loss_x, dl_dx	
end


print("------------------------Created evalFunction------------------------\n\n")

sgd_params = {
	learningRate = 0.01;
	learningRateDecay = 0;
	weightDecay = 0;
	momentum =0;
}


epochs  =3000


print("------------------------Started Training------------------------\n\n")



for i = 1,epochs do
	current_loss = 0
	for j =1,(#dataset_inputs)[1] do
		_,fs = optim.sgd(feval, x, sgd_params)
		current_loss = current_loss +fs[1]
	end
end

current_loss = current_loss / (#dataset_inputs)[1] --average loss
print ("average", current_loss)





test_loaded = csvigo.load('test.csv');

test_females = torch.DoubleTensor(test_loaded.female)
test_ages = torch.DoubleTensor(test_loaded.age)
test_ages = (test_ages - torch.min(test_ages))/ (torch.max(test_ages) - torch.min(test_ages))

test_dataset_inputs =  torch.DoubleTensor((#test_ages)[1],2)
test_dataset_inputs[{{},1}] = test_females
test_dataset_inputs[{{},2}] = test_ages

loss_x = model:forward(test_dataset_inputs)

prob = torch.exp(loss_x)
print(prob)
