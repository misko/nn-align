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
  local out = torch.DoubleTensor(n,4,1,l):zero()
  for i=1,n do
    local s = seq:narrow(1,r[i],l)
    for j=1,l do
      if s[j]>0 then
        out[i][s[j]][1][j]=1
      end
    end
    --out[i][1][1]:copy(s)
    if s:min()==0 then
      r[i]=0
    end
    assert(r[i]<seq:size()[1])
  end
  return out,r:double()/seq:size()[1]
end

function make_model() 
  local m = nn.Sequential();
  local s = 9
  local h = 256
  m:add(nn.SpatialConvolution(4, h, s, 1 ,1 ,1 ))
  m:add(nn.ReLU())
  m:add(nn.SpatialBatchNormalization(h))
  for i=1,10 do
    m:add(nn.SpatialConvolution(h, h, s, 1 ,1 ,1 ))
    m:add(nn.ReLU())
    m:add(nn.SpatialBatchNormalization(h))
  end
  m:add(nn.View(h*12))
  m:add(nn.Linear(h*12,h*4))
  m:add(nn.ReLU())
  m:add(nn.BatchNormalization(h*4))
  m:add(nn.Linear(h*4,h*4))
  m:add(nn.ReLU())
  m:add(nn.BatchNormalization(h*4))
  m:add(nn.Linear(h*4,h*4))
  m:add(nn.ReLU())
  m:add(nn.BatchNormalization(h*4))
  m:add(nn.Linear(h*4,h))
  m:add(nn.ReLU())
  m:add(nn.BatchNormalization(h))
  m:add(nn.Linear(h,h))
  m:add(nn.ReLU())
  m:add(nn.BatchNormalization(h))
  m:add(nn.Linear(h,1))

  print(m) 
  --m.criterion = nn.MSECriterion()
  m.criterion = nn.AbsCriterion()
  --m.criterion = nn.DistKLDivCriterion()
  --m.criterion = nn.SmoothL1Criterion()
  return m
end

--convert fasta to float
--seq = process_fasta('chr20.fa')
--torch.save('chr20.t7',seq)

--lets load the float
seq = torch.load('chr20.t7')
m=make_model():cuda()

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


for i=1,1000000 do
  local d,t = sample(seq,100,0,mb_sz)
  d=d:cuda()
  t=t:cuda()
  mn=0
  local feval = function(x)
      if x ~= parameters then
        parameters:copy(x)
      end
      gradParameters:zero()
      predictions = m:forward(d)
      --print(d:size(),predictions:size(),t:size())
      --t:resizeAs(predictions)
      cost = m.criterion:forward(predictions,t)
      df_do = m.criterion:backward(predictions,t)
      --print(predictions)
      mn=((predictions-t):abs():mean())
      m:backward(d,df_do) 
      return cost,gradParameters
  end
  _,f=optim.rmsprop(feval, parameters, m.rmsState)
  --_,f=optim.adam(feval, parameters, m.admState)
  --print(t:max(),t:min())
  print(string.format("%d\t%0.5e\t%.5e\t%0.3f\t%0.3f\t%0.3f\t%0.3f", seq:size()[1],mn,cost*seq:size()[1],t:mean(),t:var(),predictions:mean(),predictions:var()))
end
