import rama_py
opts = rama_py.multicut_solver_options("PD")
out = rama_py.rama_cuda([0, 1, 2], [1, 2, 0], [1.1, -2, 3], opts) 
print(out) 
