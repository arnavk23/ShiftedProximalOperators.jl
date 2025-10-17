export ShiftedNuclearnorm

mutable struct ShiftedNuclearnorm{
  R <: Real,
  S <: AbstractArray,
  T,
  Tr,
  M <: AbstractArray{T},
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} <: ShiftedProximableFunction
  h::Nuclearnorm{R, S, T, Tr, M}
  xk::V0  # base shift (nonzero when shifting an already shifted function)
  sj::V1  # current shift
  sol::V2   # internal storage
  shifted_twice::Bool
  xsy::V2
  function ShiftedNuclearnorm(
    h::Nuclearnorm{R, S, T, Tr, M},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    shifted_twice::Bool,
  ) where {R <: Real, S <: AbstractArray, T, Tr, M <: AbstractArray{T}}
    sol = similar(xk)
    xsy = similar(xk)
    new{R, S, T, Tr, M, typeof(xk), typeof(sj), typeof(sol)}(h, xk, sj, sol, shifted_twice, xsy)
  end
end

fun_name(ψ::ShiftedNuclearnorm) = "shifted Nuclearnorm"
fun_expr(ψ::ShiftedNuclearnorm) = "t ↦ Nuclearnorm(xk + sj + t)"

shifted(
  h::Nuclearnorm{R, S, T, Tr, M},
  xk::AbstractVector{R},
) where {R <: Real, S <: AbstractArray, T, Tr, M <: AbstractArray{T}} =
  ShiftedNuclearnorm(h, xk, zero(xk), false)
shifted(
  ψ::ShiftedNuclearnorm{R, S, T, Tr, M, V0, V1, V2},
  sj::AbstractVector{R},
) where {
  R <: Real,
  S <: AbstractArray,
  T,
  Tr,
  M <: AbstractArray{T},
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} = ShiftedNuclearnorm(ψ.h, ψ.xk, sj, true)

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedNuclearnorm{R, S, T, Tr, M, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
) where {
  R <: Real,
  S <: AbstractArray,
  T,
  Tr,
  M <: AbstractArray{T},
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
}
  λ = ψ.h.lambda
  # ψ.sol = q + ψ.xk + ψ.sj  (in-place to avoid temporaries)
  copyto!(ψ.sol, q)
  @inbounds for i in eachindex(ψ.sol)
    ψ.sol[i] += ψ.xk[i] + ψ.sj[i]
  end
  # copy reshaped sol into A
  copyto!(ψ.h.A, reshape_array(ψ.sol, size(ψ.h.A)))
  psvd_dd!(ψ.h.F, ψ.h.A, full = false)
  # in-place positive thresholding
  @inbounds for i in eachindex(ψ.h.F.S)
    v = ψ.h.F.S[i] - λ * σ
    ψ.h.F.S[i] = v > 0 ? v : zero(v)
  end
  for i ∈ eachindex(ψ.h.F.S)
    s = ψ.h.F.S[i]
    for j = 1:size(ψ.h.A, 1)
      ψ.h.F.U[j, i] = ψ.h.F.U[j, i] * s
    end
  end
  mul!(ψ.h.A, ψ.h.F.U, ψ.h.F.Vt)
  y .= reshape_array(ψ.h.A, size(y)) .- (ψ.xk .+ ψ.sj)
  return y
end
