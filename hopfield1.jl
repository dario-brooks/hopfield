using Random
using LinearAlgebra

struct Hopfield
    num_neurons::Int
    weights::Matrix{Float64}
end

# Function to init a hopfield ntwk with random weights:
function Hopfield(num_neurons::Int)
    weights = zeros(Float64, num_neurons, num_neurons)
    return Hopfield(num_neurons, weights)
end

# Function to train a hopfield ntwk with a set of patterns:
function train!(hopfield::Hopfield, patterns::Matrix{Int})
    num_patterns, num_neurons = size(patterns)

    for i in 1:num_neurons
        for j in 1:num_neurons
            if i != j
                for p in 1:num_patterns
                    hopfield.weights[i, j] += (2*patterns[p, i] - 1)*(2*patterns[p, j] -1)
                end
                hopfield.weights[i, j] /= num_neurons
            end
        end
    end
end

# Function to update the hopfield ntwk using asynchronous update:
function update!(hopfield::Hopfield, state::Vector{Int})
    num_neurons = hopfield.num_neurons
    i = rand(1:num_neurons)
    h = dot(hopfield.weights[i, :], state)

    # Assign the state of neuron 'i' to -1 or 1 depending on h:
    state[i] = h >= 0 ? 1 : -1
end

# Function to retrieve stored patterns from the network:
function retrieve(hopfield::Hopfield, noisy_pattern::Vector{Int}, max_iters::Int)
    state = copy(noisy_pattern)
    for _ in 1:max_iters
        update!(hopfield, state)
    end
    return state
end



## EXAMPLE 1. ##

# Define params of the network:
num_neurons = 100
num_patterns = 5
pattern_size = num_neurons

# Generate some random binary patterns:
patterns = rand(0:1, (num_patterns, pattern_size)) .|> x -> x*2 - 1

# Initialize a hopfield ntwk:
hopnet = Hopfield(num_neurons)

# Train the network:
train!(hopnet, patterns)

# Create a noisy pattern (to test for retrieval) by copying a binary pattern we created 
# earlier and then flipping some bits:
noisy_pattern = copy(patterns[1, :])
noisy_pattern[1:(pattern_size ÷ 5)] .= -noisy_pattern[1:(pattern_size ÷ 5)]

# Use the hopfield ntwk to retrieve a binary pattern from the noisy one we created:
retrieved_pattern = retrieve(hopnet, noisy_pattern, 100)

println("Original Pattern:    ", patterns[1, :])
println("Noisy Pattern:    ", noisy_pattern)
println("Retrieved Pattern:    ", retrieved_pattern)

# Cool. Now, functions to test the hamming dist and the cosine similarity between
# two patterns, e.g. the original pattern and the retrieved pattern:
function hammingdistance(pattern1::Vector{Int}, pattern2::Vector{Int})
    if length(pattern1) != length(pattern2)
        throw(ArgumentError("Pattern lengths are not equal. Patterns must have equal lengths."))
    else
        hamming_distance = sum(pattern1 .!= pattern2)
    end
    return hamming_distance
end

hammingdistance(patterns[1, :], retrieved_pattern)

function cosinesimilarity(pattern1::Vector{Int}, pattern2::Vector{Int})
    if length(pattern1) != length(pattern2)
        throw(ArgumentError("Pattern lengths are not equal. Patterns must have equal lengths."))
    else
        cosine_similarity = dot(pattern1, pattern2)/(norm(pattern1)*norm(pattern2))
    end
    return cosine_similarity
end

cosinesimilarity(patterns[1, :], retrieved_pattern)



## EXAMPLE 2: Stochastic update rule, initialize network with a random state. ##

function update2!(hopfield::Hopfield, state::Vector{Int})
    num_neurons = hopfield.num_neurons
    i = rand(1:num_neurons)
    h = dot(hopfield.weights[i, :], state)
    
    # Randomly flip the state of neuron 'i' with probability p:
    p = 1/(1 + exp(-2*h))
    state[i] = rand() < p ? 1 : -1
end

function retrieve2(hopfield::Hopfield, noisy_pattern::Vector{Int}, max_iters::Int)
    state = copy(noisy_pattern)
    for _ in 1:max_iters
        update2!(hopfield, state)
    end
    return state
end

function random_state(num_neurons::Int)
    return rand(0:1, num_neurons) .|> x -> x*2 - 1
end

num_neurons = 100
num_patterns = 5
pattern_size = num_neurons

patterns = rand(0:1, (num_patterns, pattern_size)) .|> x -> x*2 - 1

hopnet = Hopfield(num_neurons)

train!(hopnet, patterns)

noisy_pattern = copy(patterns[1, :])
noisy_pattern[1:(pattern_size ÷ 2)] .= -noisy_pattern[1:(pattern_size ÷ 2)]

retr_pattern = retrieve2(hopnet, noisy_pattern, 100)

hammingdistance(retr_pattern, patterns[1, :])
cosinesimilarity(patterns[1, :], retr_pattern)