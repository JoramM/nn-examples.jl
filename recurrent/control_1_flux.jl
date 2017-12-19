using Flux
using Flux: Tracker, mse, throttle
using Base.Iterators: repeated

function norm_input(x, min, max)
     # res = []
     res = zeros(size(x))
     for i=1:length(res)
          res[i] = (x[i]-min)/(max-min)
          # push!(res,[(x[i]-min)/(max-min)])
     end
     return res
end

function extract_output(y, min, max)
     res = deepcopy(y)
     for i=1:length(y)
          res[i] = y[i]*((max-min)+min)
     end
     return res
end

# global parameters
input_dim = 1
hidden_dim = 2
output_dim = 2
α = 0.1                 # training rate
iterations = 5000

#### dataset ####
# input (control value)
raw_input = [1 1 1 1 1 1 2 2 2 2 2 1 1 1 1]
# raw_input = [1 1 1 1 1 1 2 2 2 2 2 1 1 1 1; 1 1 1 1 1 1 2 2 2 2 2 1 1 1 1]

# output (rotation speed)
raw_labels = [15 15 15 15 15 15 30 30 30 30 30 15 15 15 15]
# raw_labels = [15 15 15 15 15 15 30 30 30 30 30 15 15 15 15; 15 15 15 15 15 15 30 30 30 30 30 15 15 15 15]

x = norm_input(raw_input,0,10)
y = norm_input(raw_labels,0,50)

dataset = collect(repeated((x,y),iterations))
# dataset = (x,y)

#### model ####

# >>> without using Flux
# W1 = param(randn(hidden_dim, input_dim))
# b1 = param(randn(hidden_dim))
# layer1(x) = W1 * x .+ b1
#     # hidden neurons in between the two layers
# W2 = param(randn(output_dim, hidden_dim))
# b2 = param(randn(hidden_dim))
# layer2(x) = W2 * x .+ b2
# model(x) = layer2(layer1(x))

# using CuArrays (GPU)
# W, b, x, y = cu.((W, b, x, y))

# >>> with Flux
model = Chain(
    Dense(input_dim, hidden_dim),
    Dense(hidden_dim, output_dim))

#### loss function ####
loss(x, y) = mse(model(x), y)

# dataset = zip(x, y)
# dataset = repeated((x, y),100)
# opt = SGD(params(m), 0.1)          # Gradient descent with learning rate
# evalcb = () -> @show(loss(x, y))   # output the progress

#### training ####
function update!(ps, η = .1)
  for w in ps
    w.data .-= w.grad .* η
    w.grad .= 0
  end
end

for i = 1:length(dataset)
    xs = dataset[i][1]
    ys = dataset[i][2]
    back!(loss(xs, ys))             # automatic backpropagation
    update!(params(model), α)
    # update!((W1, b1, W2, b2), α)

    if i%1000 == 0  # do not print every step
        @show loss(x, y)
    end
end

# Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 0.5))

# output final error
# @show(loss(x, y))

# testing the results
# @show m([1,0,0])    # should be close to 1
# @show m([0,1,0])    # should be close to 0
