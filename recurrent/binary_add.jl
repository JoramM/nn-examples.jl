using Flux
using Flux: mse, throttle, crossentropy
using Gallium
import NNlib.σ
# using Base.Iterators: repeated

# input variables
bin_length = 8
max_num = 2^bin_length

α = 0.1                     # learning rate
input_dim = 2               # two input numbers
hidden_dim = 7              # hidden neurons
output_dim = 1              # the output is sum of the two numbers
set_size = 15000               # size of the training set

# int to binary dict
int_to_bin = Dict(i => bin(i,bin_length) for i = 0:max_num)

# training data
num_x = reshape([convert(Int16,rand(0:max_num/2)) for i = 1:set_size*2], 2, set_size)
num_y = reshape([num_x[1,i] + num_x[2,i] for i = 1:set_size], 1, set_size)

# x, y = Array{Array{Float64,input_dim}}, Array{Array{Float64,output_dim}}
# x, y = [], []

# convert to bin

# for i = 1:set_size
#     a = bin(num_x[1,i],bin_length)
#     b = bin(num_x[2,i],bin_length)
#     c = bin(num_y[i],bin_length)
#
#     in_bin = zeros(2,bin_length)
#     out_bin = zeros(1,bin_length)
#     for j = 1:bin_length
#         in_bin[1,j] = convert(Float64, a[j])-48
#         in_bin[2,j] = convert(Float64, b[j])-48
#         out_bin[j] = convert(Float64, c[j])-48
#     end
#     push!(x,in_bin)
#     push!(y,out_bin)
# end

# model
# m = Chain(
#     RNN(input_dim, hidden_dim, σ),
#     Dense(hidden_dim, output_dim, σ))

function print_status(err, label, pred)
    println("Error: ", err)
    println("True: ", label)
    println("Pred: ", pred)
end

# weight matrices
syn_0 = 2*randn(input_dim,hidden_dim)
syn_1 = 2*randn(hidden_dim,output_dim)
syn_r = 2*randn(hidden_dim,hidden_dim)

syn_0_update = zeros(size(syn_0))
syn_1_update = zeros(size(syn_1))
syn_r_update = zeros(size(syn_r))

# loss function
# loss(x,y) = mse(m(x),y)
loss(x,y) = x - y
# σ(x) = 1/(1+exp.(-x))
function σ(x)
    x = 1+exp.(-x)
    for i = 1:length(x)
        x[i] = 1/x[i]
    end
    return x
end

function σ_derivative(x)
    res = 1-x
    for i = 1:length(x)
        res[i] *= x[i]
    end
    return res
end

function vecprod(a,b)
    if length(a) == length(b)
        res = reshape(zeros(length(a)),1,length(a))
        for i = 1:length(a)
            res[i] = a[i]*b[i]
        end
        return res
    end
end

# dataset = zip(x,y)
# dataset = zip(xs, ys)
# opt = SGD(params(m), α)             # Gradient descent with learning rate 0.25
# evalcb = () -> @show(loss(x, y))    # output the progress

# training
for i = 1:set_size
    in_bin_a = bin(num_x[1,i],bin_length)   # addend 1 (input 1)
    in_bin_b = bin(num_x[2,i],bin_length)   # addend 2 (input 2)
    out_bin_c = bin(num_y[i],bin_length)    # currect sum (output)

    prediction = zeros(bin_length)          # result of the net
    overallError = 0

    layer_2_Δs = []
    layer_1_values = []
    push!(layer_1_values, reshape(zeros(hidden_dim),1,hidden_dim))

    # calculate output and error
    for pos = bin_length:-1:1
        a = convert(Float64, in_bin_a[pos])-48
        b = convert(Float64, in_bin_b[pos])-48

        x = [a b]
        y = [convert(Float64, out_bin_c[pos])-48]'

        # MODEL -->
        # hidden layer (input ~+ prev_hidden)
        layer_1 = σ(x*syn_0 + layer_1_values[end]*syn_r)

        # output layer (new binary representation)
        layer_2 = σ(layer_1*syn_1)

        layer_2_error = loss(y,layer_2)
        push!(layer_2_Δs, layer_2_error*σ_derivative(layer_2))     # save derivatives for each timestep
        overallError += abs(layer_2_error[1])

        prediction[pos] = round(layer_2[1,1])

        push!(layer_1_values, deepcopy(layer_1))                   # store hidden layer for next timestep
        # MODEL --<
    end

    future_layer_1_Δ = reshape(zeros(hidden_dim),1,hidden_dim)

    # backpropagation
    for pos = 1:bin_length
        x = [convert(Float64, in_bin_a[pos])-48,convert(Float64, in_bin_b[pos])-48]
        layer_1 = layer_1_values[end-pos+1]        # selecting the current hidden layer
        prev_layer_1 = layer_1_values[end-pos]     # selecting the previous hidden layer

        # error at output layer
        layer_2_Δ = layer_2_Δs[end-pos+1]
        # error at hidden layer (given the error at the future hidden layer and the current output error)
        layer_1_Δ = vecprod((future_layer_1_Δ*syn_r' + layer_2_Δ*syn_1'),σ_derivative(layer_1))

        # let's update all our weights so we can try again
        syn_1_update += layer_1'*layer_2_Δ
        syn_r_update += prev_layer_1'*layer_1_Δ
        syn_0_update += x*layer_1_Δ

        future_layer_1_Δ = layer_1_Δ
    end

    # backpropagation is calculated, now update the weights
    syn_0 += syn_0_update * α
    syn_1 += syn_1_update * α
    syn_r += syn_r_update * α

    syn_0_update *= 0
    syn_1_update *= 0
    syn_r_update *= 0

    if i % (0.1*set_size) == 0
        print_status(overallError, out_bin_c, prediction)
    end
end

# Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 0.5))
# Flux.train!(loss, dataset, opt)

# output final error
# @show overallError

# testing the results
