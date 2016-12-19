require "nn"
require "optim"
require "cutorch"
require "cunn"

cutorch.setDevice(1)

function process_fasta(file) 
  --file='chr20.fa'
  --lines = {}
  t={}
  for line in io.lines(file) do 
    line:gsub(".",
      function(c)
        local c=c:lower() 
        local v=0
        if c=="a" then
          v=1
        elseif c=="c" then
          v=2
        elseif c=="g" then
          v=3
        elseif c=="t" then
          v=4
        end
        table.insert(t,v) 
      end)
    --lines[#lines + 1] = line
  end
  return torch.FloatTensor(t)
end

function sample(seq,l,m,n) 
  local r = (torch.rand(n)*(seq:size()[1]-l)):int()+1
  local out = torch.DoubleTensor(n,1,1,l)
  for i=1,n do
    out[i][1][1]:copy(seq:narrow(1,r[i],l))
  end
  return out,r:double()-seq:size()[1]/2
end

function make_model(l) 
  local m = nn.Sequential();
  local s = 9
  local h = 128
  m:add(nn.SpatialConvolution(1, h, s, 1 ,1 ,1 ))
  m:add(nn.ReLU())
  m:add(nn.SpatialBatchNormalization(h))
  for i=1,11 do
    m:add(nn.SpatialConvolution(h, h, s, 1 ,1 ,1 ))
    m:add(nn.ReLU())
    m:add(nn.SpatialBatchNormalization(h))
  end
  m:add(nn.View(h*4))
  m:add(nn.Linear(h*4,h*4))
  m:add(nn.ReLU())
  m:add(nn.BatchNormalization(h*4))
  m:add(nn.Linear(h*4,h))
  m:add(nn.ReLU())
  m:add(nn.BatchNormalization(h))
  m:add(nn.Linear(h,1))
  --m:add(nn.SpatialConvolution(h, h, s, 1 ,1 ,1 , 0, 0))
  --m:add(nn.SpatialConvolution(h, 1, s, 1 ,1 ,1 , 0, 0))
  m.criterion = nn.MSECriterion()
  return m
end

--convert fasta to float
--seq = process_fasta('chr20.fa')
--torch.save('chr20.t7',seq)

--lets load the float
seq = torch.load('chr20.t7')
m=make_model(20):cuda()

m:training()

m.admState = {
  epsilon = 1e-4,
  learningRate = 0.01,
  weightDecay = 0
}

m.rmsState = {
  learningRate = 0.01,
  epsilon = 0.001,
  alpha = 0.95
}

if m.parameters_flat == nil then
  m.parameters_flat, m.gradParameters_flat = m:getParameters()
end
local parameters, gradParameters = m.parameters_flat, m.gradParameters_flat

local mb_sz=64


for i=1,10000 do
  local d,t = sample(seq,100,0,mb_sz)
  d=d:cuda()
  t=t:cuda()
  local feval = function(x)
      if x ~= parameters then
        parameters:copy(x)
      end
      gradParameters:zero()
      predictions = m:forward(d)
      cost = m.criterion:forward(predictions,t)
      df_do = m.criterion:backward(predictions,t)
      m:backward(d,df_do) 
      return cost,gradParameters
  end
  --_,f=optim.rmsprop(feval, parameters, m.rmsState)
  _,f=optim.adam(feval, parameters, m.admState)
  print(cost,predictions:mean())
end
