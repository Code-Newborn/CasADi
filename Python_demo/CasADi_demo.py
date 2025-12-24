import casadi as ca

# ---- 例子：最小 NLP ----
# min (x1-1)^2 + (x2-2)^2
# s.t. g(x)= x1 + x2 = 1
x = ca.MX.sym("x", 2)  # type: ignore
f = (x[0]-1)**2 + (x[1]-2)**2
g = ca.vertcat(x[0] + x[1])

lam = ca.MX.sym("lam", g.size1())   # 约束乘子 # type: ignore
sigma = ca.MX.sym("sigma")          # 目标缩放（Hessian of Lagrangian 常用）# type: ignore

# 导数函数
grad_f = ca.gradient(f, x)
jac_g = ca.jacobian(g, x)
L = sigma*f + ca.dot(lam, g)        # 拉格朗日函数
hess_L, _ = ca.hessian(L, x)

F = ca.Function("F", [x], [f], ["x"], ["f"])
G = ca.Function("G", [x], [g], ["x"], ["g"])
GradF = ca.Function("GradF", [x], [grad_f], ["x"], ["grad_f"])
JacG = ca.Function("JacG", [x], [jac_g], ["x"], ["jac_g"])
HessL = ca.Function("HessL", [x, lam, sigma], [hess_L], ["x", "lam", "sigma"], ["hess_L"])

# ---- 生成 C 代码（单个函数也可直接 .generate）----
cg_opts = {"with_header": True}
cg = ca.CodeGenerator("nlp_funcs", cg_opts)
cg.add(F)
cg.add(G)
cg.add(GradF)
cg.add(JacG)
cg.add(HessL)
cg.generate()
print("Generated: nlp_funcs.c / nlp_funcs.h")
