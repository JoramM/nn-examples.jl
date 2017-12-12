using Flux
using Flux: mse, throttle, crossentropy
# using Gallium
# using Base.Iterators: repeated

# input variables
bin_length = 8
max_num = 2^bin_length

α = 0.1                     # learning rate
input_dim = 2               # two input numbers
hidden_dim = 7              # hidden neurons
output_dim = 1              # the output is sum of the two numbers
set_size = 1000               # size of the training set

# int to binary dict
int_to_bin = Dict(i => bin(i,bin_length) for i = 0:max_num)

# training data
num_x = reshape([convert(Int16,rand(0:max_num/2)) for i = 1:set_size*2], 2, set_size)
num_y = reshape([num_x[1,i] + num_x[2,i] for i = 1:set_size], 1, set_size)

# x, y = Array{Array{Float64,input_dim}}, Array{Array{Float64,output_dim}}
x, y = [], []

# convert to bin

for i = 1:set_size
    a = bin(num_x[1,i],bin_length)
    b = bin(num_x[2,i],bin_length)
    c = bin(num_y[i],bin_length)

    in_bin = zeros(2,bin_length)
    out_bin = zeros(1,bin_length)
    for j = 1:bin_length
        in_bin[1,j] = convert(Float64, a[j])-48
        in_bin[2,j] = convert(Float64, b[j])-48
        out_bin[j] = convert(Float64, c[j])-48
    end
    push!(x,in_bin)
    push!(y,out_bin)
end

# model
m = Chain(
    RNN(input_dim, hidden_dim, σ),
    Dense(hidden_dim, output_dim, σ))

# loss function
loss(x,y) = mse(m(x),y)
# loss(x,y) = m(x) - y

# xs = [[[0 0 0 0 0 0 0 1;0 0 0 0 0 0 1 0]],
#       [[0 0 0 0 0 0 0 1;0 0 0 0 0 0 1 0]],
#       [[0 0 0 0 0 0 0 1;0 0 0 0 0 0 1 0]]]
#
# ys = [[[0 0 0 0 0 0 1 1]],
#       [[0 0 0 0 0 0 1 1]],
#       [[0 0 0 0 0 0 1 1]]]

dataset = zip(x,y)
# dataset = zip(xs, ys)
opt = SGD(params(m), α)             # Gradient descent with learning rate 0.25
evalcb = () -> @show(loss(x, y))    # output the progress

# for d in dataset
#     @show l = loss(d...)
# end

# Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 0.5))
Flux.train!(loss, dataset, opt)

# output final error
@show(loss([0 0 0 0 0 0 0 1; 0 0 0 0 0 0 1 0], [0 0 0 0 0 0 1 1]))

# testing the results
