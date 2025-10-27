export ShiftedNormL1B2

mutable struct ShiftedNormL1B2{
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} <: ShiftedProximableFunction
  h::NormL1{R}
  xk::V0
  sj::V1
  sol::V2
  Δ::R
  χ::NormL2{R}
  shifted_twice::Bool
  xsy::V2

  function ShiftedNormL1B2(
    h::NormL1{R},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    Δ::R,
    χ::NormL2{R},
    shifted_twice::Bool,
  ) where {R <: Real}
    sol = similar(sj)
    xsy = similar(sj)
    new{R, typeof(xk), typeof(sj), typeof(sol)}(h, xk, sj, sol, Δ, χ, shifted_twice, xsy)
  end
end

@inline function _chi_norm(χ::NormL2{R}, v::AbstractVector{R}) where {R}
  s2 = zero(R)
  @inbounds for i in eachindex(v)
    vi = v[i]
    s2 += vi * vi
  end
  return χ.lambda * sqrt(s2)
end

function (ψ::ShiftedNormL1B2)(y)
  @inbounds for i in eachindex(ψ.xsy)
    ψ.xsy[i] = ψ.xk[i] + ψ.sj[i] + y[i]
  end
  s2 = zero(eltype(y))
  @inbounds for i in eachindex(y)
    v = ψ.sj[i] + y[i]
    s2 += v * v
  end
  χy = ψ.χ.lambda * sqrt(s2)
  ball_val = (χy <= ψ.Δ) ? zero(χy) : oftype(χy, Inf)
  return ψ.h(ψ.xsy) + ball_val
end

shifted(h::NormL1{R}, xk::AbstractVector{R}, Δ::R, χ::NormL2{R}) where {R <: Real} =
  ShiftedNormL1B2(h, xk, zero(xk), Δ, χ, false)
shifted(
  ψ::ShiftedNormL1B2{R, V0, V1, V2},
  sj::AbstractVector{R},
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} =
  ShiftedNormL1B2(ψ.h, ψ.xk, sj, ψ.Δ, ψ.χ, true)

fun_name(ψ::ShiftedNormL1B2) = "shifted L1 norm with L2-norm trust region indicator"
fun_expr(ψ::ShiftedNormL1B2) = "t ↦ ‖xk + sj + t‖₁ + χ({‖sj + t‖₂ ≤ Δ})"
fun_params(ψ::ShiftedNormL1B2) =
  "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14 * "Δ = $(ψ.Δ)"

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedNormL1B2{R, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
  λ = ψ.λ
  λχ = ψ.χ.lambda

  function projB!(dest::AbstractVector{R}, scale::R)
    @inbounds for i in eachindex(dest)
      lo = ψ.sj[i] + q[i] - λ * σ
      hi = ψ.sj[i] + q[i] + λ * σ
      zi = -(ψ.xk[i]) * scale
      dest[i] = zi < lo ? lo : (zi > hi ? hi : zi)
    end
    return dest
  end

  function chi_norm(v::AbstractVector{R})
    s2 = zero(R)
    @inbounds for i in eachindex(v)
      vi = v[i]
      s2 += vi * vi
    end
    return λχ * sqrt(s2)
  end

  projB!(y, one(R))

  if ψ.Δ <= chi_norm(y)
    froot(η) = begin
      scale = η / ψ.Δ
      projB!(ψ.sol, scale)
      η - chi_norm(ψ.sol)
    end

    f0 = froot(zero(R))
    fΔ = froot(ψ.Δ)
    eta = zero(R)
    if f0 == zero(R)
      eta = zero(R)
    elseif fΔ == zero(R)
      eta = ψ.Δ
    elseif f0 * fΔ < zero(R)
      eta = find_zero(froot, (zero(R), ψ.Δ), Roots.Bisection())
    else
      η0 = ψ.Δ / 2
      eta = try
        find_zero(froot, η0)
      catch e
        @warn "Root finding failed: $e; falling back to Δ" exception=(e, catch_backtrace())
        ψ.Δ
      end
    end

    if eta == zero(R)
      @inbounds for i in eachindex(y)
        lo = ψ.sj[i] + q[i] - λ * σ
        hi = ψ.sj[i] + q[i] + λ * σ
        yi = zero(R)
        y[i] = yi < lo ? lo : (yi > hi ? hi : yi)
      end
    else
      scale = eta / ψ.Δ
      projB!(y, scale)
      s = ψ.Δ / eta
      @inbounds for i in eachindex(y)
        y[i] *= s
      end
    end
  end

  @inbounds for i in eachindex(y)
    y[i] -= ψ.sj[i]
  end

  χy = chi_norm(y)
  if χy > ψ.Δ
    s = ψ.Δ / χy
    @inbounds for i in eachindex(y)
      y[i] *= s
    end
  end
  return y
end
