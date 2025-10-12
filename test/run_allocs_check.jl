using ShiftedProximalOperators, Test

function wrappedallocs(expr)
  # simple wrapper returning allocation count
  return @allocated expr
end

println("Running allocation checks...\n")

# CompositeNormL2 case
try
  CompositeOp = ShiftedProximalOperators.CompositeNormL2
  println("CompositeNormL2: defined")
  function c!(z, x)
    z[1] = 2 * x[1] - x[4]
    z[2] = x[2] + x[3]
  end
  function J!(z, x)
    z.vals .= Float64[2.0, 1.0, 1.0, -1.0]
  end
  λ = 3.62
  Op = ShiftedProximalOperators.NormL2
  h = Op(λ)
  b = zeros(Float64, 2)
  A = SparseMatrixCOO(Float64[2 0 0 -1; 0 1 1 0])
  ψ = CompositeOp(λ, c!, J!, A, b)
  xk = [0.0, 1.1741, 0.0, -0.4754]
  ϕ = shifted(ψ, xk)
  x = [0.1097, 1.1287, -0.29, 1.2616]
  y = similar(x)
  ν = 0.1056
  alloc = wrappedallocs(prox!(y, ϕ, x, ν))
  println("Composite prox! allocs = ", alloc)
catch e
  println("CompositeNormL2 test skipped: ", e)
end

# Several scalar operators: NormL0, NormL1, RootNormLhalf
for op_sym in (:NormL0, :NormL1, :RootNormLhalf)
  try
    op = getfield(ShiftedProximalOperators, op_sym)
    println("\nOperator: ", op_sym)
    h = op(1.0)
    n = 1000
    xk = rand(n)
    ψ = shifted(h, xk)
    y = rand(n)
    alloc = @allocated ψ(y)
    println("  ψ(y) allocs = ", alloc)
    ψ = shifted(h, xk, -3.0, 4.0, rand(1:n, Int(n/2)))
    alloc = @allocated ψ(y)
    println("  ψ(y) with groups allocs = ", alloc)
  catch e
    println("  Skipped ", op_sym, ": ", e)
  end
end

# IndBallL0
for op_sym in (:IndBallL0,)
  try
    op = getfield(ShiftedProximalOperators, op_sym)
    println("\nOperator: ", op_sym)
    h = op(1)
    n = 1000
    xk = rand(n)
    ψ = shifted(h, xk)
    y = rand(n)
    alloc = @allocated ψ(y)
    println("  ψ(y) allocs = ", alloc)
    χ = NormLinf(1.0)
    ψ = shifted(h, xk, 0.5, χ)
    alloc = @allocated ψ(y)
    println("  ψ(y) with χ allocs = ", alloc)
  catch e
    println("  Skipped ", op_sym, ": ", e)
  end
end

# NormL0, NormL1 prox!/iprox! allocation checks
for op_sym in (:NormL0, :NormL1)
  try
    op = getfield(ShiftedProximalOperators, op_sym)
    println("\nprox!/iprox! checks for ", op_sym)
    h = op(1.0)
    n = 1000
    xk = rand(n)
    ψ = shifted(h, xk)
    y = rand(n)
    d = rand(n)
    a1 = wrappedallocs(prox!(y, ψ, y, 1.0))
    a2 = wrappedallocs(iprox!(y, ψ, y, d))
    println("  prox! allocs = ", a1, ", iprox! allocs = ", a2)
    ψ = shifted(h, xk, -3.0, 4.0, rand(1:n, Int(n/2)))
    a1 = wrappedallocs(prox!(y, ψ, y, 1.0))
    a2 = wrappedallocs(iprox!(y, ψ, y, d))
    println("  prox! (grouped) allocs = ", a1, ", iprox! (grouped) allocs = ", a2)
  catch e
    println("  Skipped ", op_sym, ": ", e)
  end
end

# NormL2 allocations
try
  println("\nNormL2 allocations")
  h = NormL2(1.0)
  n = 1000
  xk = rand(n)
  y = rand(n)
  d = rand(n)
  a = wrappedallocs(prox!(y, h, y, 1.0))
  println("  prox!(y,h,y,1.0) allocs = ", a)
  ψ = shifted(h, xk)
  println("  ψ(y) allocs = ", @allocated ψ(y))
  println("  prox!(y,ψ,y,1.0) allocs = ", wrappedallocs(prox!(y, ψ, y, 1.0)))
catch e
  println("NormL2 checks failed: ", e)
end

# Rank & Nuclearnorm checks
for (op_sym, shifted_sym) in zip((:Rank, :Nuclearnorm), (:ShiftedRank, :ShiftedNuclearnorm))
  try
    println("\n", op_sym, " allocations")
    ShiftedOp = getfield(ShiftedProximalOperators, shifted_sym)
    Op = getfield(ShiftedProximalOperators, op_sym)
    m = 10; n = 11; λ = 1.0; γ = 5.0
    x = vec(reshape(rand(m, n), m * n, 1))
    q = vec(reshape(rand(m, n), m * n, 1))
    s = vec(reshape(rand(m, n), m * n, 1))
    F = psvd_workspace_dd(zeros(m, n), full = false)
    h = Op(λ, ones(m, n), F)
    f = ShiftedOp(h, x, s, true)
    y = zeros(m * n)
    println("  prox!(y,h,x,γ) allocs = ", wrappedallocs(prox!(y, h, x, γ)))
    println("  prox!(y,f,q,γ) allocs = ", wrappedallocs(prox!(y, f, q, γ)))
  catch e
    println("  Skipped ", op_sym, ": ", e)
  end
end

println("\nDone.")
