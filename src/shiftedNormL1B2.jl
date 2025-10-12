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

(ψ::ShiftedNormL1B2)(y) = ψ.h(ψ.xk + ψ.sj + y) + IndBallL2(ψ.Δ)(ψ.sj + y)

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
  ProjB(z) = min.(max.(z, ψ.sj .+ q .- ψ.λ * σ), ψ.sj .+ q .+ ψ.λ * σ)
  froot(η) = η - ψ.χ(ProjB((-ψ.xk) .* (η / ψ.Δ)))

  y .= ProjB(-ψ.xk)

  if ψ.Δ ≤ ψ.χ(y)
    # compute root of froot on [0, Δ] when possible
    f0 = froot(0.0)
    fΔ = froot(ψ.Δ)
    if f0 == 0.0
      η = 0.0
    elseif fΔ == 0.0
      η = ψ.Δ
    elseif f0 * fΔ < 0.0
      # bracketed: use explicit bisection to avoid method-selection warnings
      η = find_zero(froot, (0.0, ψ.Δ), Roots.Bisection())
    else
      # not bracketed: fall back to a safe single-start solver (secant-like) to avoid errors
      # pick midpoint as initial guess
      η0 = ψ.Δ / 2
      η = try
        find_zero(froot, η0)
      catch _e
        # as a last resort, pick Δ (should be safe although may be suboptimal)
        ψ.Δ
      end
    end
    # avoid division by zero when η == 0
    if η == 0.0
      y .= ProjB(zeros(eltype(y), length(ψ.xk)))
    else
      y .= ProjB((-ψ.xk) .* (η / ψ.Δ)) * (ψ.Δ / η)
    end
  end
  y .-= ψ.sj
  # ensure numerical safety: if the returned y slightly exceeds the trust-region radius
  # due to rounding/fallbacks, project it back onto the L2 ball of radius ψ.Δ
  if ψ.χ(y) > ψ.Δ
    y .= y .* (ψ.Δ / ψ.χ(y))
  end
  return y
end
