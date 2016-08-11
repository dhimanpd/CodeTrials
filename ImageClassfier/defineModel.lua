require 'cunn'

model=nn.Sequential()

model:add(nn.SpatialConvolution(1,300, 3, 3))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
model:add(nn.SpatialDropout(0))

-- stage 2 : filter bank -> squashing -> max pooling
model:add(nn.SpatialConvolution(300, 300, 2, 2))
model:add(nn.Tanh())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
model:add(nn.SpatialDropout(0))

 -- stage 3 : standard 2-layer neural network
model:add(nn.Reshape(300*2*30))
model:add(nn.Linear(300*2*30, 100))
model:add(nn.Tanh())
model:add(nn.Linear(100,58))

