module HPOMDPs

export
    HPOMDP,
    evaluate_XADD

using XADDs, SymPy

mutable struct HPOMDP
	g::XADDGraph
	state_discrete::Array{Bool, 1}
	state_continuous::Array{Float64, 1}
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

end