module HPOMDPs

export
    HPOMDP,
    HPOMDP_State,
    evaluate_XADD,
    substitution_var,
    substitution_xadd

using XADDs, SymPy

mutable struct HPOMDP_State
	state_continuous::{Array{Float64,1}}
	state_discrete::{Array{Int, 1}}
end

mutable struct HPOMDP
	g::XADDGraph
	num_actions::Int
	obs_discrete_probs::Array{Int, 1} #Array of XADD nodes for discete observation probabilities
	obs_continuous_probs::Array{Int, 1} #Arrray of XADD nodes for continuous observations - should output observation not probability
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

function backup(problem::HPOMDP, alphas::Array{Int}, belief::Array{Tuple{Float64, HPOMDP_State}})
	for a in 1:problem.num_actions
		#TODO - Deal With Continuous Observations, for now assume observations are discrete
		gamma_a = Array{Int, 1}
		obs_set = Array{Array{Int}}
		get_obs_set(problem, Array{Int}[], obs_set)
		for o in obs_set

		end
	end
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
