require 'Csv'
require 'cunn'
require 'optim'
csvData  = Csv('featureVector.csv','r')


data = csvData:readall()
tensorData = torch.DoubleTensor(data);
output = tensorData:select(2,61)
input = tensorData[{{},{1,60}}]

traininginput = torch.DoubleTensor(2160,60)
trainingoutput = torch.DoubleTensor(2160)

testinginput = torch.DoubleTensor(540,60)
testingoutput = torch.DoubleTensor(540)

count = 1;
traincount= 1;
testcount= 1;
for i =1,2700 do
	if count >40 then
		testinginput[testcount]= input[i]
		testingoutput[testcount]= output[i]
		testcount = testcount+1;
	else
		traininginput[traincount]= input[i]
		trainingoutput[traincount]= output[i]
		traincount = traincount+1;
	end
	if count == 50 then
		count = 0;
	end
	count= count +1;
end


ninputs= 60
nhidden = 90
noutputs = 54
linlayer = nn.Linear(ninputs,nhidden)

softMaxLayer = nn.LogSoftMax()
model = nn.Sequential()
model:add(nn.Reshape(ninputs))

model:add(linlayer)
model:add(nn.Tanh())
model:add(nn.Linear(nhidden,noutputs))
--model:add(nn.Tanh())

criterion = nn.ClassNLLCriterion()

x, dl_dx = model:getParameters()
print("------------------------Model Created------------------------\n\n")
print("------------------------Creating evalFunction------------------------\n\n")

feval = function(x_new)
   if x ~= x_new then
	x:copy(x_new)
   end
   _nidx_ = (_nidx_ or 0) + 1
   
   if _nidx_ > ((#traininginput)[1]) then _nidx_ = 1 end
   local inputs = traininginput[_nidx_]
   local target = trainingoutput[_nidx_]
   dl_dx:zero()				--Gradient Zero

	mdlforward  = model:forward(traininginput)
   local loss_x = criterion:forward(mdlforward,trainingoutput)
   model:backward(traininginput, criterion:backward(mdlforward, trainingoutput))
	return loss_x, dl_dx	
end


print("------------------------Created evalFunction------------------------\n\n")

sgd_params = {
	learningRate = 0.1;
	learningRateDecay = 0;
	weightDecay = 0;
	momentum =0;
}


epochs  =3000


print("------------------------Started Training------------------------\n\n")



for i = 1,epochs do
	current_loss = 0
	_,fs = optim.sgd(feval, x, sgd_params)
	current_loss = current_loss +fs[1]
	print(fs[1])
end

