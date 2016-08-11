require 'optim'
require 'cunn'
require 'Csv'
require 'cutorch'
if Features ==nil then
	dofile("dataload.lua")
end
dofile("defineModel.lua") 

criterion = nn.MSECriterion()
model:cuda()
criterion:cuda()
parameters, gradParameters = model:getParameters()

currentLearningRate=0.1
currentepoch=0
epochs=50

-- SGD Optimization Algo
print("Optimization Algo - SGD")
sgd_params = {
   	learningRate = 0.01,
   	learningRateDecay = 0,
   	weightDecay = 0,
   	momentum = 0, 
}


batchSize =200
for i = 1,epochs do
	currentepoch=i   
	local trainError = 0
	local testError = 0
	local AccurateCount =0
	local AccurateTop3Count=0
	local AccurateTop5Count=0
	outFileData = torch.DiskFile('Output.txt','w')
	for t = 1,(#Features)[1],batchSize do
           
      -- create mini batch
		local inputs = {}
		local targets = {}
		for i = t,math.min(t+batchSize-1,(#Features)[1]) do
         -- load new sample
	        local filename =string.format(name_format,ImageA[i][1])
            local images_set=image.load("images/" .. filename,1)
            images_set=image.scale(images_set,500,50)
            images_set:resize(1,50,500)
            local input =images_set:cuda()
			local target = Features[i]
			table.insert(inputs, input)
			table.insert(targets, target)
		end
		local feval = function(x)
         -- get new parameters
			if x ~= parameters then
				parameters:copy(x)
			end

         -- reset gradients
			gradParameters:zero()

         -- f is the average of all criterions
			local f = 0

         -- evaluate function for complete mini batch
			for i = 1,#inputs do
				-- estimate f
			--	local output =  model:forward(images_set)
				local output = model:forward(inputs[i])
				local err = criterion:forward(output,  targets[i])
				f = f + err

				-- estimate df/dW
				local df_do = criterion:backward(output, targets[i])
				model:backward(inputs[i], df_do)
			end

         -- normalize gradients and f(X)
			gradParameters:div(#inputs)
			f = f/#inputs
			trainError = trainError + f

			 -- return f and df/dX
			return f,gradParameters
		end

		_,fs = optim.sgd(feval,parameters,sgd_params)
  
      --Calculate Accuracy
   --   print("Accuracy Calculate "  .. t )
	   if currentepoch%25 ==0 then
	        for i = t,math.min(t+batchSize-1,(#Features)[1]) do
	         -- load new sample
		        local filename =string.format(name_format,ImageA[i][1])
	            local images_set=image.load("images/" .. filename,1)
	          
	            images_set=image.scale(images_set,500,50)
	            images_set:resize(1,50,500)
	            output =  model:forward(images_set:cuda())
			    ImOut =torch.DoubleTensor((#FeatureA)[1]):zero()
			    for k=1,(#FeatureA)[1] do
					local FGAccFeatures=torch.DoubleTensor((#FeatureA)[2]):zero()
					FGAccFeatures = Features[k]:double()-meanA[1]-output:double():cmul(stddevA[1])
					FGAccFeatures = (FGAccFeatures:cdiv(sigma[1])-FGMean[1]):cdiv(FGStdDev[1])
					ImOut[k] =nn.Sigmoid():cuda():forward(FGModel:forward(FGAccFeatures:cuda())):double()
				end
				sorted_value,Sorted_index =torch.sort(ImOut)
				if ImageA[i][1]==ImageA[Sorted_index[1]][1] then
					AccurateCount=AccurateCount+1
				end
				else
				   outFileData:writeString(string.format("Failed %d=[%d,%d,%d,%d,%d],[%0.10f,%0.10f,%0.10f,%0.10f,%0.10f]\n",ImageA[i][1],ImageA[Sorted_index[1]][1],ImageA[Sorted_index[2]][1],ImageA[Sorted_index[3]][1],ImageA[Sorted_index[4]][1],ImageA[Sorted_index[5]][1],sorted_value[1],sorted_value[2],sorted_value[3],sorted_value[4],sorted_value[5]))
				end
			end
		end
	end  
	outFileData:close()
	trainError = trainError / math.floor((#Features)[1])
	print(string.format("currentepoch=%d",currentepoch))
    if currentepoch%25 ==0 then
	    print(string.format("trainError=%f, Top Accuracy =%f,count=%d, Top 3 Accuracy =%f,count=%d, Top 5 Accuracy=%f,count=%d",trainError,100*AccurateCount/(#Features)[1],AccurateCount,100*AccurateTop3Count/(#Features)[1],AccurateTop3Count,100*AccurateTop5Count/(#Features)[1],AccurateTop5Count))
	end
end