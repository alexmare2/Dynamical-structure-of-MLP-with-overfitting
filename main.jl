using Flux, Optimisers
using Random, Distributions, Plots
using QuadGK  # Integration
using LaTeXStrings  # Use Latex to print plot titles
using BenchmarkTools  # Time functions
using LinearAlgebra  # Eigenvalues of the jacobian

JULIA_NUM_OF_THREADS = 60  # Number of threads to use

# Define generic variables
DATA_FILE = "data.dat"
IMG_FOLDER = pwd()*"/tmp_img/"
RESUME = false
ORBITS = false
SHOW = true
SAVE = true
ACTION = "main"  # "main", "save", "research_orbit", "test", "eigval_plot", "check_number_eig"
RAND_SEED = 1724  #set to 1724 for reproducibility
overfitted = [[0.5479504133334038; 2.154742368932136;;], [1.6105171890567702 0.6711825665103458]]
initial = [[-0.13; 0.13;;], [0.16 -0.15]]

# Initialise instance variables
sample_size = 100
teacher_size = 1
student_size = 2
non_linearity = tanh
MAX_IT = 200 #3000000
N_OF_NETWORKS = 1
data_noise_var = 0.0
learning_step = 0.01
n_of_graphs = 0  #IF 0, it doesnt produce GIFS
if n_of_graphs > 0  # IF 0 it will not make GIFs
    timestamp_list = [convert(Int32, floor(MAX_IT / (MAX_IT ^ (t / n_of_graphs))))
                    for t in range(1, convert(Int32, floor(log(MAX_IT) / log(MAX_IT ^ (1 / n_of_graphs)))))]
    push!(timestamp_list, MAX_IT - 1)
else
    timestamp_list = []
end
func_evolution = Animation()
orbit_evolution = Animation()
trajectories = [[] for _ in 1:N_OF_NETWORKS]

# Define useful functions
normal = Normal(0, 1)
data_noise_distr = Normal(0, data_noise_var)
param_distr_1 = Uniform(-2.5,2.5)
param_distr_2 = Uniform(-2.5,2.5)
dertan(z) = 1 - tanh(z)^2
derdertan(z) = -2 * tanh(z) * dertan(z)
function grad_2(network, j, k, sample, train_set)
    -2 * tanh(network[1].weight[j] * sample[k]) * (train_set[k] - network([sample[k]])[1])
end
function grad_1(network, j, k, sample, train_set)
    -2 * sample[k] * network[2].weight[j] * dertan(network[1].weight[j] * sample[k]) *
    (train_set[k] - network([sample[k]])[1])
end

# Initialise model
Random.seed!(RAND_SEED) # Setting the seed
train_E = []
gen_E = []
graph_x = -5.0:0.1:5.0  # x limits of the plots 

teacher_neurons = [hcat(1.0), hcat(2.0)]  #Initialise teacher
teacher = Chain(
    Dense(teacher_neurons[1], zeros(teacher_size), non_linearity),
    Dense(teacher_neurons[2], [0])
)

random_init_neurons = [initial] #[[rand(param_distr_2,student_size,1), rand(param_distr_1,1,student_size)] for _ in 1:(N_OF_NETWORKS)]   #Initialise student
student_theta = random_init_neurons[1]
student = Chain(
    Dense(student_theta[1], zeros(student_size), non_linearity),
    Dense(student_theta[2], [0])
)

initial_theta = deepcopy(student_theta)
init_stud = Chain(
    Dense(initial_theta[1], zeros(student_size), non_linearity),
    Dense(initial_theta[2], [0])
)

#Generate sample data
if !isfile(DATA_FILE)
    println("Generating data...")
    sample = rand(normal, sample_size)
    train_set = []
    for x in sample
        append!(train_set, (teacher([x]) .+ rand(data_noise_distr)))
    end
else
    println("Reading data from $DATA_FILE...")
    sample = Float64[]
    train_set = Float64[]
    open(DATA_FILE, "r") do io
        for line in eachline(io)
            x, y = parse.(Float64, split(line))
            push!(sample, x)
            push!(train_set, y)
        end
    end
end

eval_teacher(x::Float64) = teacher([x])[1]
eval_student(x::Float64) = student([x])[1]
eval_init_student(x::Float64) = init_stud([x])[1]

# Define main functions
function save_trjs()
    open("new_traj.txt", "w") do io
        for i in 1:length(trajectories[1])
            println(io, "$(10*i) $(trajectories[1][i])")
        end
    end
end

function read_trjs()
    traj = []
    open("traj_no_noise.txt", "r") do io
        for line in eachline(io)
            push!(traj, eval(Meta.parse.(split(line, "0 ")[2])))
        end
    end
    return [traj]
end

function plot_functions()
    img = plot(eval_teacher, graph_x, xlimits = (-4,4), ylimits = (-2,2), linestyle = :dash, color = "red", xlabel="", ylabel="", label = "", dpi=600, title="")
    plot!(eval_student, graph_x, color = "blue", label = "")
    plot!(eval_init_student, graph_x, color = "blue", linestyle = :dash, label = "")
    scatter!(sample, train_set, color = "pink", markershape = :star5, markerstrokewidth=0, label = "")
    return img
end

function plot_specials(quale)
    if quale == "w"
        optimals = [[0., 0, -1, 1, 1, -1], [1., -1, 0, 0, 1, -1]]
        ovf= [[overfitted[1][1], overfitted[1][2], -overfitted[1][1], -overfitted[1][2], overfitted[1][1], overfitted[1][2], -overfitted[1][1], -overfitted[1][2]],
        [overfitted[1][2], overfitted[1][1], -overfitted[1][2], -overfitted[1][1], -overfitted[1][2], -overfitted[1][1], overfitted[1][2], overfitted[1][1]]]
        scatter!(optimals[1], optimals[2], color = "lightgray", markerstrokewidth=0, label = "", markersize=4, markershape = :star5)

        scatter!(ovf[1], ovf[2], color = "red", markerstrokewidth=0, label = "", markersize=6, markershape = :star4)

        plot!(y -> 0, color="gray", label = "")
        plot!(y -> y, color="gray", label = "")
        plot!([0,0],[-10,10], color="gray", label = "")
        plot!(y -> 1, color="lightgray", label = "", linewidth=1.7)
        plot!(y -> -1, color="lightgray", label = "", linewidth=1.7)
    elseif quale == "v"
        optimals = [[0., 0, 2, -2, ], [2., -2,0, 0, 1, -1]]
        ovf= [[overfitted[2][1], overfitted[2][2], -overfitted[2][1], -overfitted[2][2], overfitted[2][1], overfitted[2][2], -overfitted[2][1], -overfitted[2][2]],
        [overfitted[2][2], overfitted[2][1], -overfitted[2][2], -overfitted[2][1], -overfitted[2][2], -overfitted[2][1], overfitted[2][2], overfitted[2][1]]]
        scatter!(optimals[1], optimals[2], color = "lightgray", markerstrokewidth=0, label = "", markersize=4, markershape = :star5)

        scatter!(ovf[1], ovf[2], color = "red", markerstrokewidth=0, label = "", markersize=6, markershape = :star4)

        plot!(y -> 0, color="gray", label = "")
        plot!([-10,10],[0,0], color="gray", label = "") 
        plot!(y -> y, color="gray", label = "")
        plot!(y -> 2-y, color="lightgray", label = "", linewidth=1.7)
        plot!(y -> 2+y, color="lightgray", label = "", linewidth=1.7)
        plot!(y -> -2-y, color="lightgray", label = "", linewidth=1.7)
        plot!(y -> -2+y, color="lightgray", label = "", linewidth=1.7)        
    end

end

function plot_parameters(quale)
    set = [[student[1].weight], [student[2].weight]]  # [w], [v]
    
    colours = ["blue", "red", "green", "purple", "orange", "yellow", "pink", "brown"]
    for t in random_init_neurons
        for i in 1:2
                push!(set[i], t[i])
        end
    end

    if quale == "w"
        p1 = plot(xlimits = (-3,3), ylimits = (-2.5,2.5), color = "red", xlabel="", ylabel="", legend=:topright, legendfontsize=6, title = "", dpi=600)
    elseif quale == "v"
        p1 = plot(xlimits = (-3,3), ylimits = (-2.5,2.5), color = "red", xlabel="", ylabel="", legend=:topright, legendfontsize=6, title = "", dpi=600)
    end
    plot_specials(quale)
    scatter!([100], [100], color = "darkblue", markerstrokewidth=0, label = "", markersize=1.8)
    #for j in 1:student_size
    #    scatter!(set[j][2], set[j][1], color = colours[j], markerstrokewidth=0, label = "", markersize=1)
    #    scatter!([set[j][2][1]], [set[j][1][1]], color = "darkblue", markerstrokewidth=0, label="", markersize=1.8)
    #end

    return p1
end

function plot_traj()
    try  #Choose the number for the image
        if !isempty(readdir(IMG_FOLDER))
            global img_number = maximum([parse(Int32, split(img,"_")[end][1:end-4]) for img in readdir(IMG_FOLDER)])+1
        else
            global img_number = 0
        end
    catch e
        global img_number = 999
        println(e)
        println("Enumeration conflict. Please check data in the folder.")
    end

    S = []
    for k in 1:N_OF_NETWORKS
        set = [[[],[]] for _ in 1:student_size]
        for t in trajectories[k]
            for i in 1:2
                for j in 1:student_size
                    push!(set[j][i], t[i][j])
                end
            end
        end
        push!(S, deepcopy(set))
    end

    # prepare data for both subplots (assumes at least one network)
    xs_w = S[1][1][1]
    ys_w = S[1][2][1]
    xs_v = S[1][1][2]
    ys_v = S[1][2][2]

    #times = collect(1:10:MAX_IT)  # time index for coloring (assume same length for both)
    #log_times = log10.(times)        # log-scale time mapping
    times = 1:length(xs_w)
    log_times = log10.(times)

    # create the two base parameter plots
    p_w = plot_parameters("w")
    p_v = plot_parameters("v")

    # combine into a single figure with two subplots side-by-side,
    # give a single shared title and force a consistent output size
    # clear the individual subplot titles (they were set in plot_parameters)
    lyt = @layout([a b c{0.08w}])
    # create a tiny invisible third subplot to host the colorbar so subplot index 3 exists
    p_cb = plot(legend = false, xlim = (0, 1), ylim = (0, 1), framestyle = :none, xticks = false, yticks = false)
    plt = plot(p_w, p_v, p_cb, layout = lyt, size = (1200, 600), title = "")

    scatter!(plt, xs_w, ys_w, marker_z = log_times, c = :viridis, markersize = 2.7, markerstrokewidth = 0, label = "", subplot = 1, colorbar = false)
    scatter!(plt, xs_v, ys_v, marker_z = log_times, c = :viridis, markersize = 2.7, markerstrokewidth = 0, label = "", subplot = 2, colorbar = false)

    # colorbar-only subplot (use tiny/invisible markers to produce colorbar)
    scatter!(plt, [0.0, 1.0], [0.0, 1.0],
             marker_z = [minimum(log_times), maximum(log_times)],
             c = :viridis, markersize = 0.1, markerstrokewidth = 0, label = "",
             subplot = 3, colorbar = true, clims = (minimum(log_times), maximum(log_times)),
             xticks = false, yticks = false, framestyle = :none)
    # save or display the single combined figure
    if !SAVE && SHOW
        display(plt)
    elseif SAVE
        savefig(plt, IMG_FOLDER * "trajectory_both_$img_number.png")
    end

    return plt
end

function Jacobian(theta_in)
    theta = []
    for i in 1:student_size
        push!(theta, theta_in[2][i])
        push!(theta, theta_in[1][i])
    end

    function jf2(x,i)
        if i%2==1
            return tanh(theta[i+1]*x)
        elseif i%2==0
            return x*theta[i-1]*dertan(theta[i]*x)
        end
    end
    function Jf1(k,i,j) 
        net = Chain(
            Dense(theta_in[1], zeros(student_size), non_linearity),
            Dense(theta_in[2], [0])
        )
        eval_net(x) = net([x])[1]
        if i==j && i%2==0
            return sample[k]^2*theta[i-1]*derdertan(theta[i]*sample[k])*(eval_net(sample[k])-train_set[k])
        elseif i==j+1 && i%2==0
            return sample[k]*dertan(theta[i]*sample[k])*(eval_net(sample[k])-train_set[k])
        elseif i==j-1 && i%2==1
            return sample[k]*dertan(theta[i+1]*sample[k])*(eval_net(sample[k])-train_set[k])
        else
            return 0
        end
    end
    function Jf2(k,i,j)
        return jf2(sample[k],i)*jf2(sample[k],j)
    end
    jf(k) = [(-Jf1(k, i, j) - Jf2(k, i, j)) for i in 1:2*student_size, j in 1:2*student_size]
    Jf = sum(jf(k) for k in 1:sample_size)/sample_size
    return Jf
end

function eigval(theta)
    return eigvals(Jacobian(theta))  # Returns eigenvalues and eigenvectors of the Jacobian matrix
end

function eigvec(theta)
    vecs= []
    for v in eachrow(eigvecs(Jacobian(theta)))
        v1 = zeros(student_size,1)
        v2 = zeros(1,student_size)
        for i in 1:2*student_size
            if i%2==1
                v2[div(1,2)+1] = v[i]
            else
                v1[div(i, 2)] = v[i]
            end
        end
        push!(vecs, [v1,v2])
    end
    return vecs
end

function gradient(network, sample, train_set)
    total_grad = [zeros(student_size,1), zeros(1,student_size)]
    for i in 1:sample_size
        for j in 1:student_size
            total_grad[1][j] += grad_1(network, j, i, sample, train_set)
            total_grad[2][j] += grad_2(network, j, i, sample, train_set)
        end
    end
    return total_grad ./ sample_size
end

function train_epoch(network)
    total_grad = [zeros(student_size,1),zeros(1,student_size)]
    total_err = 0
    for i in 1:sample_size #threads --> SGD
        for j in 1:student_size
            total_grad[1][j] += grad_1(network, j, i, sample, train_set)
            total_grad[2][j] += grad_2(network, j, i, sample, train_set)
        end
        total_err += (eval_student(sample[i]) - train_set[i])^2
    end
    #println(total_grad./sample_size)
    return total_err/sample_size, total_grad./sample_size
end

function SGD_train_epoch(network)
    total_grad = [zeros(student_size,1),zeros(1,student_size)]
    total_err = 0
    Threads.@threads for i in 1:sample_size #threads --> SGD
        for j in 1:student_size
            total_grad[1][j] += grad_1(network, j, i, sample, train_set)
            total_grad[2][j] += grad_2(network, j, i, sample, train_set)
        end
        total_err += (eval_student(sample[i]) - train_set[i])^2
    end
    return 0.1*total_err/sample_size, total_grad./sample_size
end

function theo_err(teacher, student)
    I = [0,0]
    function integrand(x)
        return (1/sqrt(2*pi))*exp(-x^2/2)*(teacher([x])[1]-student([x])[1])^2
    end
    try
        I = quadgk(integrand, -Inf, Inf, rtol=1e-3)  #returns [I1,I2]: 1=result, 2=approx error
    catch
        return 1e6
    end
    if I[1]>1e-16 && I[1]<1e6
        return I[1]
    elseif I[1]>1e6
        println("Whoops, something went wrong...")
        quit()
    else 
        println("Theoretical optimal is achieved!!")
        return 0.00000005
    end
end 

function MC_check_optimal_eig(n)
    points = []
    for _ in 1:floor(n/3)
        r = rand(2)
        #r /= sum(r)
        r *= 2
        r = round.(r, digits=5)
        point = [[0.0 r[2]]', [r[1] 0]]  #[[1.0 1.0]', [r[1] r[2]]]
        push!(points, point)
    end

    for _ in 1:floor(n/3)
        r = rand(2)
        #r /= sum(r)
        r *= 2
        r = round.(r, digits=5)
        point = [[0.0 0.0]', [r[1] r[2]]]  #[[1.0 1.0]', [r[1] r[2]]]
        push!(points, point)
    end

    for _ in 1:floor(n/3)
        r = rand(2)
        #r /= sum(r)
        r *= 2
        r = round.(r, digits=5)
        point = [[r[1] r[2]]', [0.0 0.0]]  #[[1.0 1.0]', [r[1] r[2]]]
        push!(points, point)
    end

    count_0 = 0
    count_1 = 0
    count_2 = 0
    count_3 = 0
    count_4 = 0
    Threads.@threads for point in points
        e = eigval(point)
        pos_count = count(x -> x > 0, e)
        if pos_count == 0
            count_0 += 1
        elseif pos_count == 1
            count_1 += 1
        elseif pos_count == 2
            count_2 += 1
        elseif pos_count == 3
            count_3 += 1
        elseif pos_count == 4
            count_4 += 1
        end
    end
    println("Out of $n points:")
    println("0 positive eigenvalues: $count_0")
    println("1 positive eigenvalues: $count_1")
    println("3 positive eigenvalues: $count_3")
    println("2 positive eigenvalues: $count_2")
    println("4 positive eigenvalues: $count_4")
end

function MC_check_plateau_eig(n)
    points = []
    for i in 1:n
        r1 = rand(student_size, 1)
        r2 = rand(1, student_size)
        r1 = r1*0.0001/sum(r1)
        r2 = r2*0.0001/sum(r2)
        push!(points, [overfitted[1]+r1, overfitted[2]+r2])
    end
    count_0 = 0
    count_1 = 0
    count_2 = 0
    count_3 = 0
    count_4 = 0
    Threads.@threads for point in points
        e = eigval(point)
        pos_count = count(x -> x > 0, e)
        if pos_count == 0
            count_0 += 1
        elseif pos_count == 1
            count_1 += 1
        elseif pos_count == 2
            count_2 += 1
        elseif pos_count == 3
            count_3 += 1
        elseif pos_count == 4
            count_4 += 1
        end
    end
    println("Out of $n points:")
    println("0 positive eigenvalues: $count_0")
    println("1 positive eigenvalues: $count_1")
    println("3 positive eigenvalues: $count_3")
    println("2 positive eigenvalues: $count_2")
    println("4 positive eigenvalues: $count_4")
end

function plot_eigval()
    N = length(random_init_neurons)
    init_colors = [RGB(0,0,0) for _ in 1:N]
    c1 = RGB(0,1,0)
    c2 = RGB(1,0,1)
    c3 = RGB(1,0,0)
    c4 = RGB(0,0,1)
    c5 = RGB(0,1,1)
    c6 = RGB(1,153/255,51/255)
    c7 = RGB(153/255,0,153/255)

    p1 = plot(xlimits = (-5,5), ylimits = (-1,8), color = "red", xlabel=L"v_i", ylabel=L"w_j", legend=true, legendfontsize=6, title = "Hessian eigenvalues classification", dpi=600)
    plot_specials()
    scatter!([16], [15], color = c1, markerstrokewidth=0, label="0 positive eigenvalues")
    scatter!([16], [15], color = c2, markerstrokewidth=0, label="1 positive eigenvalues")
    scatter!([16], [15], color = c3, markerstrokewidth=0, label="2 positive eigenvalues")
    scatter!([16], [15], color = c4, markerstrokewidth=0, label="3 positive eigenvalues")
    scatter!([16], [15], color = c5, markerstrokewidth=0, label="4 positive eigenvalues")
    scatter!([16], [15], color = c6, markerstrokewidth=0, label="5 positive eigenvalues")
    scatter!([16], [15], color = c7, markerstrokewidth=0, label="6 positive eigenvalues")
    scatter!([16], [15], color = RGB(0,0,0), markerstrokewidth=0, label="7 or 8 positive eigenvalues")

    Threads.@threads for i in 1:N
        e = eigval(random_init_neurons[i])
        pos_count = count(x -> x > 0, e)
        if pos_count == 0
            init_colors[i] = c1
        elseif pos_count == 1
            init_colors[i] = c2
        elseif pos_count == 2
            init_colors[i] = c3
        elseif pos_count == 3
            init_colors[i] = c4
        elseif pos_count == 4
            init_colors[i] = c5
        elseif pos_count == 5
            init_colors[i] = c6
        elseif pos_count == 6
            init_colors[i] = c7
        end
    end
    for i in 1:N
        for j in 1:student_size
            scatter!([random_init_neurons[i][2][j]], [random_init_neurons[i][1][j]], color = init_colors[i], markerstrokewidth=0, markersize=1, label="")
        end
    end
    return p1
end

println("Instance initialised correctly. Training the Neural network.")
global epoch = 0

# Train the neural network

function main()
    while epoch < MAX_IT
        epoch += 1
        global student, trajectories, epoch
        for j in 1:N_OF_NETWORKS
            t=random_init_neurons[j]
            model = Chain(
                Dense(t[1], zeros(student_size), non_linearity),
                Dense(t[2], [0])
            )
            if epoch%10 == 1
                push!(trajectories[j], [deepcopy(student[1].weight), deepcopy(student[2].weight)])
            end
        end
        err, grad = train_epoch(student)
        push!(train_E, err)
        push!(gen_E, theo_err(teacher, student))

        if n_of_graphs > 0 && N_OF_NETWORKS > 0
            Threads.@threads for i in 1:N_OF_NETWORKS
                t=random_init_neurons[i]
                model = Chain(
                    Dense(t[1], zeros(student_size), non_linearity),
                    Dense(t[2], [0])
                )
                temp_grad = gradient(model, sample, train_set)
                global random_init_neurons[i] -= learning_step.*temp_grad
            end
        end

        if epoch in timestamp_list
            plot_functions()
            frame(func_evolution)
            plot_parameters()
            frame(orbit_evolution)
        end

        student_theta[1] .-= learning_step .* grad[1]
        student_theta[2] .-= learning_step .* grad[2]
        student = Chain(
            Dense(student_theta[1], zeros(student_size), non_linearity),
            Dense(student_theta[2], [0])
        )
    end

    @show initial_theta
    @show student_theta
    println("Training completed! Plotting results.")
end

function plot_result()
    # Plots
    try  #Choose the number for the image
        if !isempty(readdir(IMG_FOLDER))
            global img_number = maximum([parse(Int32, split(img,"_")[end][1:end-4]) for img in readdir(IMG_FOLDER)])+1
        else
            global img_number = 0
        end
    catch e
        global img_number = 999
        println(e)
        println("Enumeration conflict. Please check data in the folder.")
    end

    @show img_number
    
    #Plot function evolution
    img = plot_functions()
    if SAVE
        savefig(IMG_FOLDER*"plot_$img_number.png")
    end
    if SHOW
        display(img)
    end

    #Plot training and Generalization Error
    img = plot(train_E, color="black", xaxis=:log, yaxis=:log, label="", xlabel="", ylabel="", dpi=600, title="")
    plot!(gen_E, color="black", xaxis=:log, yaxis=:log, label="", linestyle=:dash)
    if SAVE
        savefig(IMG_FOLDER*"err_$img_number.png")
    end
    if SHOW
        display(img)
    end

    # Create GIF
    if n_of_graphs > 0 && SAVE
        gif(func_evolution, IMG_FOLDER*"Function_Evolution_$img_number.gif", fps=5)
        gif(orbit_evolution, IMG_FOLDER*"Orbits_Evolution_$img_number.gif", fps=5)
    end

    println("Results plotted correctly.")
end


if ACTION == "main"
    @time main()  #Benchmark the main function
    plot_result()
    plot_traj()
    #save_trjs()  #Uncomment to save trajectories to a file
elseif ACTION == "read"
    println("Reading files.")
    trajectories = read_trjs()
    MAX_IT = length(trajectories)
    epoch = MAX_IT
    main()  #Benchmark the main function
    plot_traj()
elseif ACTION == "save"
    # Save the sample and train_set to a data file
    open(DATA_FILE, "w") do io
        for i in 1:sample_size
            println(io, "$(sample[i]) $(train_set[i])")
        end
    end
    println("Data saved to $DATA_FILE.")
elseif ACTION == "research_orbit"
    if N_OF_NETWORKS == 0
        println("N_OF_NETWORKS is set to 0. Please set it to a positive number.")
        exit()
    elseif N_OF_NETWORKS > 30
        println("N_OF_NETWORKS is set to a very high number. Please set it to a lower number.")
        exit()
    end
    random_init_neurons = []
    for i in 1:N_OF_NETWORKS
        q1 = randn(student_size, 1)
        q1 *= 0.05/sum(q1)
        q2 = randn(1, student_size)
        q2 *= 0.05/sum(q2)
        perturbation = [q1, q2]
        push!(random_init_neurons, student_theta+perturbation)
    end
    student_theta = random_init_neurons[1]
    student = Chain(
    Dense(student_theta[1], zeros(student_size), non_linearity),
    Dense(student_theta[2], [0])
    )

    initial_theta = deepcopy(student_theta)
    init_stud = Chain(
        Dense(initial_theta[1], zeros(student_size), non_linearity),
        Dense(initial_theta[2], [0])
    )
    @time main()
    plot_traj()
    savefig(IMG_FOLDER*"trajectory_$img_number.png")
elseif ACTION == "eigval_plot"
    push!(random_init_neurons, optimal_4n)
    push!(random_init_neurons, overfitted)
    plot_eigval()
    savefig(IMG_FOLDER*"eigvals_$img_number.png")
    println("Eigenvalue plot saved correctly.")
elseif ACTION == "check_number_eig"
    println("Checking number of eigenvalues...")
    MC_check_optimal_eig(10000)
elseif ACTION == "test"
    SAVE = false
    println("Testing the model... IMAGES ARE NOT SAVING")
    @time main()
    plot_result()
    plot_traj()
else
    println("Invalid action. Please choose 'main', 'save', 'research_orbit', 'test' or 'eigval_plot'.")
    exit()
end
