using Knet: Knet, minibatch, param, param0, softmax, nll, RNN
using Pickle
using NPZ
using Statistics: mean
using DataStructures: OrderedDict

# Load data
dir = "..\\dataset\\representations\\uncond\\cp\\ailab17k_from-scratch_cp"
t2i, i2t = Pickle.load(open("$(dir)\\dictionary.pkl"))
train = NPZ.npzread("$(dir)\\train_data_linear.npz")
test = NPZ.npzread("$(dir)\\test_data_linear.npz")

toktids = OrderedDict("tempo"=>1,"chord"=>2,"bar-beat"=>3,"type"=>4,"pitch"=>5, "duration"=>6, "velocity"=>7)
n_tokens = [length(t2i[k]) for (k, v) in toktids]

t2i # For reference

# Model settings
BATCH_SIZE = 25
EMBED_SIZES = [256, 256, 64, 32, 512, 128, 128]
D_MODEL = 512
N_HEAD = 8

# Minibatching
train_x = trunc.(Int, permutedims(
          cat(train["x"], reshape(train["mask"], (size(train["x"],1),size(train["x"],2),1)), dims=3),
          [2, 3, 1]).+1);                                           @show size(train_x) # T, K+1, B
train_y = trunc.(Int, permutedims(train["y"], [2, 3, 1]).+1);       @show size(train_y) # T, K, B
test_x = trunc.(Int, permutedims(
         cat(test["x"], reshape(test["mask"], (size(test["x"],1),size(test["x"],2),1)), dims=3),
         [2, 3, 1]).+1);                                            @show size(test_x) # T, K+1, B
test_y = trunc.(Int, permutedims(test["y"], [2, 3, 1]).+1);         @show size(test_y) # T, K, B

train_loader = minibatch(train_x, train_y, BATCH_SIZE; shuffle=true)
test_loader = minibatch(test_x, test_y, BATCH_SIZE; shuffle=true)

length.((train_loader, test_loader))

# Simple useful layers

struct Linear; W; b; end
Linear(input::Int, output::Int) = Linear(param(output, input), param0(output))
(l::Linear)(x) = l.W*x .+ l.b

struct Embedding; W; end
Embedding(n_tokens::Int, embed::Int) = Embedding(param(embed, n_tokens))
(e::Embedding)(x) = e.W[:, x]

# TODO: Linear Transformer backbone
# ϕ(x) = elu(x) + 1 
# struct Transformer; W_Q; W_K; W_V; ff; end
# Transformer(n_layers, n_heads, q_dim, v_dim, ff_dim; activation=ϕ, dropout=0.1) =
#     Transformer(param(q_dim, ???), param(k_dim, ???), param(v_dim, ???), ff_dim)

# Sampling function for predicting tokens
function sampling(x; dims=1) # TODO : Temperature
    # size(x) = (N_tokens[type], B, T)
    x = softmax(x, dims=dims)
    first.(Tuple.(argmax(x, dims=dims))) # TODO: Weighted sampling
end

struct CPTransformer; embeds; lin_in; lin_transformer; projs; blend_type; end

CPTransformer(n_tokens::Vector{Int}, embed_sizes::Vector{Int}, d_model::Int, d_inner::Int;
    blend_dim=32) =
    CPTransformer([Embedding(n, e) for (n, e) in zip(n_tokens, embed_sizes)],
            Linear(sum(embed_sizes), d_model),
            RNN(d_model, d_model), # Placeholder until Transformer implementation
            [Linear(d_model, n) for n in n_tokens],
            Linear(d_model + blend_dim, d_model))

# y    => y != nothing ? [training mode] : [interference mode]
# gen  => gen ? return ŷ : return ŷ_P
function (model::CPTransformer)(x; y=nothing, gen=false)
    x, mask = x[:, 1:end-1, :], x[:, end, :];                    @show size(x) # (T, N_tkn, B)
    
    x = vcat([embed(x[:, i, :]) for (embed, i) in
            zip(model.embeds, 1:length(model.embeds))]...);      @show size(x) # (X_emb, T, B)
    
    x = cat([model.lin_in(x[:,i,:]) for i in
            1:size(x, 2)]..., dims=3);                           @show size(x) # (X_in, B, T)
    
    # x = Positional_Encoding(x) <-- TODO
    
    h = model.lin_transformer(x);                                @show size(h) # (D_m, B, T)
    
    ŷ_type_P = (cat([model.projs[toktids["type"]](h[:,:,i])
            for i in 1:size(h, 3)]..., dims=3));                 @show size(ŷ_type_P) # (N_tvoc, B, T)
    
    ŷ_type = y!=nothing ? y[:, toktids["type"], :] : 
            reshape(sampling(ŷ_type_P), (size(ŷ_type_P, 3), :)); @show size(ŷ_type) # (T, B)
               
    ŷ_τ = vcat([permutedims(h, [1,3,2]), 
            model.embeds[toktids["type"]](ŷ_type)]...);          @show size(ŷ_τ) # (D_m + blend, B, T)
    
    h_ = cat([model.blend_type(ŷ_τ[:, i, :]) for i in
            1:size(ŷ_τ, 2)]..., dims=3);                         @show size(h_) # (D_m, B, T)
    
    ŷ_P = [permutedims(i!=toktids["type"] ? 
            cat([proj(h[:,:,i]) for i in 1:size(h, 3)]..., dims=3) : 
            ŷ_type_P, [3,1,2]) for (proj,i)
            in zip(model.projs, 1:length(model.projs))];         @show size.(ŷ_P) # (T, N_tvoc, B)*    
    
    gen ? hcat([permutedims(sampling(permutedims(P, [2,3,1])),[3,1,2]) for P in ŷ_P]...) : ŷ_P
end

function (model::CPTransformer)(x, y; train=true)
    ŷ_P = train ? model(x, y=y) : model(x)
    loss = mean([nll(ŷ_P[i], reshape(y[:,i,:], (size(y,1),1,:)), dims=2) for i in length(n_tokens)])
end

model = CPTransformer(n_tokens, EMBED_SIZES, 512, 2048)
x, y = first(train_loader)
loss = model(x, y)
@show loss

x, y = first(test_loader)
gen = model(x, gen=true)
@show summary.((y, gen))


