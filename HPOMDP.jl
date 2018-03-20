include("/home/jdieter/CS239/project/XADDs.jl/src/XADDs.jl")
module HPOMDPs

export
    HPOMDP,
    HPOMDP_State,
    evaluate_XADD,
    substitution_var,
    substitution_xadd

using XADDs, SymPy

mutable struct HPOMDP_State
	state_continuous::Array{Float64,1}
	state_discrete::Array{Int, 1}
end

#For XADDs, variables must be initalized with names xs1,xs2,...,ds1,ds2,..., xo1,xo2,...,do1,do2,...,a,xsp1,xsp2,...,dsp1,dsp2,...
mutable struct HPOMDP
	g::XADDGraph
	gamma::Float64
	num_actions::Int
	obs_discrete_probs::Array{Int, 1} #Array of XADD nodes for discete observation probabilities
#	obs_continuous_probs::Array{Int, 1} #Arrray of XADD nodes for continuous observations - should output observation not probability
    state_transitions_cont::Array{Int, 1} #Array of XADD nodes for continuous state transitions - should output next state not probability
    state_transitions_discrete::Array{Int, 1} #Array of XADD nodes for discrete state transitions - should output probability
    rewards::Int #XADD node for rewards
end

function evaluate_XADD(g::XADDGraph, root_node::Int, vars::Dict)
	node = g.id_node_map[root_node]
	exp = node.value
	value = N(subs(exp, vars), 15)
	if g.id_node_map[root_node].true_dest_id == -1
		return value
	end

	if (node.equality && value == 0) || (!node.equality && value >= 0)
		return evaluate_XADD(g, node.true_dest_id, vars)
	else
		return evaluate_XADD(g, node.false_dest_id, vars)
	end
end

#Substitutes a variable with an XADD
function substitution_xadd(g::XADDGraph, variable, sub_node::Int, root_node::Int)
	if g.id_node_map[sub_node].true_dest_id == -1
		return substitution_var(g, variable, g.id_node_map[sub_node].value, root_node)
	end
	true_tree = substitution_xadd(g, variable, g.id_node_map[sub_node].true_dest_id, root_node)
	false_tree = substitution_xadd(g, variable, g.id_node_map[sub_node].false_dest_id, root_node)
	return add_decision_node!(g, true_tree, false_tree, g.id_node_map[sub_node].value, equality=g.id_node_map[sub_node].equality)
end

#Substitutes a variable with an expression
function substitution_var(g::XADDGraph, variable, expression, root_node::Int)
	new_value = subs(g.id_node_map[root_node].value, variable, expression)
	if g.id_node_map[root_node].true_dest_id == -1
		return add_terminal_node!(g, new_value)
	end
	true_tree = substitution_var(g, variable, expression, g.id_node_map[root_node].true_dest_id)
	false_tree = substitution_var(g, variable, expression, g.id_node_map[root_node].false_dest_id)
	return add_decision_node!(g, true_tree, false_tree, new_value, equality=g.id_node_map[root_node].equality)
end

function backup(problem::HPOMDP, alphas::Array{Tuple{Int, Int}})
	g = problem.g
	Gamma = Tuple{Int, Int}[]
	for a in 1:problem.num_actions
		#TODO - Deal With Continuous Observations, for now assume observations are discrete
		gamma_a = Int[]
		obs_set = Array{Int}[]
		state_set = Array{Int}[]
		get_obs_set(problem, Int[], obs_set)
		get_state_set(problem, Int[], state_set)
		cross_sum = [add_terminal_node!(g,0)]
		for o in obs_set
			probo = add_terminal_node!(g,1)
			for j in 1:length(problem.obs_discrete_probs)
				if o[j] == 1
					probo = apply!(g, probo, problem.obs_discrete_probs[j], :prod)
				else
					onestub = add_terminal_node!(g,1)
					notproboj = apply!(g, onestub, problem.obs_discrete_probs[j], :minus)
					probo = apply!(g, probo, notproboj, :prod)
				end
			end

			g_hao = Int[]
			for alph_prev in alphas
				alpha_prev_p = alph_prev[1]
				for i in 1:length(problem.state_transitions_cont)
					alpha_prev_p = substitution_var(g, g.symbol_map["xs$i"], g.symbol_map["xsp$i"], alpha_prev_p)
					alpha_prev_p = substitution_var(g, g.symbol_map["ds$i"], g.symbol_map["dsp$i"], alpha_prev_p)
				end
				g_haoj = add_terminal_node!(g, 0)
				for s in state_set
					g_summand = probo
					for j in 1:length(problem.state_transitions_discrete)
						if s[j] == 1
							g_summand = apply!(g, g_summand, problem.state_transitions_discrete[j], :prod)
						else
							onestub = add_terminal_node!(g,1)
							notprobsj = apply!(g, onestub, problem.state_transitions_discrete[j], :minus)
							g_summand = apply!(g, onestub, notprobsj, :prod)
						end							
					end
					g_summand = apply!(g, g_summand, alpha_prev_p, :prod)
					for j in 1:length(problem.state_transitions_cont)
						g_summand = substitution_xadd(g, g.symbol_map["xsp$j"], problem.state_transitions_cont[j], g_summand)
						g_summand = reduce!(g, g_summand)
					end
					for j in 1:length(problem.state_transitions_discrete)
						g_summand = substitution_var(g, g.symbol_map["dsp$j"], s[j], g_summand)
					end
					g_haoj = apply!(g, g_haoj, g_summand, :add)
				end
				g_haoj = substitution_var(g, g.symbol_map["a"], a, g_haoj)
				push!(g_hao, g_haoj)
			end
			cross_sum = compute_cross_sum(g, cross_sum, g_hao)
		end
		R_a = substitution_var(g, g.symbol_map["a"], a, problem.rewards)
		Gamma_a = Int[]
		for vec in cross_sum
			newvec = apply!(g, add_terminal_node!(g, problem.gamma), vec, :prod)
			newvec = apply!(g, newvec, R_a, :add)
			push!(Gamma_a, newvec)
		end
		for g_a in Gamma_a
			push!(Gamma, (g_a, a))
		end
	end
	return Gamma
end

function compute_cross_sum(g::XADDGraph, set1::Array{Int}, set2::Array{Int})
	result = Int[]
	for s1 in set1
		for s2 in set2
			push!(result, apply!(g, s1, s2, :add))
		end
	end
	return result
end

function get_obs_set(problem::HPOMDP, leading_array::Array{Int}, set::Array{Array{Int}})
	onearr = copy(leading_array)
	zeroarr = copy(leading_array)
	push!(onearr, 1)
	push!(zeroarr, 0)
	if length(onearr) == length(problem.obs_discrete_probs)
		push!(set, onearr)
		push!(set, zeroarr)
	else
		get_obs_set(problem, onearr, set)
		get_obs_set(problem, zeroarr, set)
	end
end

function get_state_set(problem::HPOMDP, leading_array::Array{Int}, set::Array{Array{Int}})
	onearr = copy(leading_array)
	zeroarr = copy(leading_array)
	push!(onearr, 1)
	push!(zeroarr, 0)
	if length(onearr) == length(problem.state_transitions_discrete)
		push!(set, onearr)
		push!(set, zeroarr)
	else
		get_state_set(problem, onearr, set)
		get_state_set(problem, zeroarr, set)
	end
end

function backup_belief(problem::HPOMDP, alphas::Array{Tuple{Int, Int}}, beliefs::Array{Array{Tuple{HPOMDP_State, Float64}, 1}})
	g = problem.g
	new_alphas = backup(problem, alphas)
	belief_alphas = Tuple{Int, Int}[]
	for b in beliefs
		max_value = -10000000000
		max_alpha = new_alphas[1]
		for alpha in new_alphas
			value = 0
			for s in b
				vars = Dict()
				for i in 1:length(problem.state_transitions_discrete)
					vars[g.symbol_map["ds$i"]] = s[1].state_discrete[i]
				end
				for i in 1:length(problem.state_transitions_cont)
					vars[g.symbol_map["xs$i"]] = s[1].state_continuous[i]
				end
				value += evaluate_XADD(g, alpha[1], vars)*s[2]
			end
			print(value)
			if (value > max_value)
				max_value = value
				max_alpha = alpha
			end
		end
		push!(belief_alphas, max_alpha)
	end
	return belief_alphas
end

g = XADDGraph()
xs1 = add_variable!(g, "xs1")
xsp1 = add_variable!(g, "xsp1")
add_variable!(g, "ds1")
add_variable!(g, "dsp1")
a = add_variable!(g, "a")

state_transitions_discrete = [add_terminal_node!(g, 1)]
n1 = add_terminal_node!(g, xs1 - 5)
n2 = add_terminal_node!(g, xs1 + 7)
state_transitions_cont =[add_decision_node!(g, n1, n2, a - 1, equality=true)]

n1 = add_terminal_node!(g, 0.9)
n2 = add_terminal_node!(g, 0.1)
n3 = add_decision_node!(g, n2, n1, xsp1 - 15)
n4 = add_terminal_node!(g, 0.7)
n5 = add_terminal_node!(g, 0.3)
n6 = add_decision_node!(g, n5, n4, xsp1 - 15)
obs_discrete_probs = [add_decision_node!(g, n3, n6, a-1, equality=true)]

n1 = add_terminal_node!(g, -1000)
n2 = add_terminal_node!(g, 100)
n3 = add_decision_node!(g, n1, n2, xs1 - 15)
n4 = add_terminal_node!(g, -1)
rewards = add_decision_node!(g, n4, n3, a-1, equality=true)

problem = HPOMDP(g, 0.95, 2, obs_discrete_probs, state_transitions_cont, state_transitions_discrete, rewards)

state1 = HPOMDP_State([16], [1])
state2 = HPOMDP_State([5], [1])
state3 = HPOMDP_State([10], [1])

belief = [(state1, 1./3.), (state2, 1./3.), (state3, 1./3.)]
belief2 = [(state3, 1./2.), (state2, 1./2.)]
belief3 = [(state2, 1.)]
alphas = [(add_terminal_node!(g,0), 1)]

for i in 1:10
	new_alphas = backup_belief(problem, alphas, [belief, belief2, belief3])
	draw_graph(g, new_alphas[1][1], filename="alpha_1_$i")
	draw_graph(g, new_alphas[2][1], filename="alpha_2_$i")
	draw_graph(g, new_alphas[3][1], filename="alpha_3_$i")
	alphas = new_alphas
end

end