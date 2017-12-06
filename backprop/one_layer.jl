using Flux
using Flux: mse, throttle
using Base.Iterators: repeated

# input
x = [0 0 1 1
     0 1 0 1
     1 1 1 1]
# output
y = [0 0 1 1]

# model
m = Dense(3,1,σ)    # only one layer (excl. input layer) with 3 inputs, 1 output and a sigmoid function

# loss function
loss(x,y) = mse(m(x), y)           # calculates the mean squared error

dataset = repeated((x, y), 10000)  # Train the model 10000 times with the same dataset
opt = SGD(params(m), 0.25)         # Gradient descent with learning rate 0.25
evalcb = () -> @show(loss(x, y))   # output the progress

Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 0.5))

# output final error
@show(loss(x, y))

# testing the results
@show m([1,0,0])    # should be close to 1
@show m([0,1,0])    # should be close to 0
