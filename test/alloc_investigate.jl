using ShiftedProximalOperators

function inspect_shifted_eval(op_sym)
  println("Inspecting ", op_sym)
  h = getfield(ShiftedProximalOperators, op_sym)(1.0)
  n = 1000
  xk = rand(n)
  ψ = shifted(h, xk)
  y = rand(n)
  println("ψ type: ", typeof(ψ))
  println("xk type: ", typeof(ψ.xk))
  # some shifted types (Box variants) have fields :selected, :sj, :xsy
  has_selected = hasfield(typeof(ψ), :selected)
  has_sj = hasfield(typeof(ψ), :sj)
  has_xsy = hasfield(typeof(ψ), :xsy)
  println("has selected: ", has_selected, ", has sj: ", has_sj, ", has xsy: ", has_xsy)

  # Measure allocation for computing the shifted vector used by ψ.h
  if has_selected && has_sj && has_xsy
    alloc1 = @allocated begin
      @. ψ.xsy = @views ψ.xk[ψ.selected] + ψ.sj[ψ.selected] + y[ψ.selected]
    end
    println("alloc for xsy assignment: ", alloc1)

    alloc2 = @allocated begin
      val = ψ.h(ψ.xsy)
    end
    println("alloc for ψ.h(ψ.xsy): ", alloc2)
  elseif has_sj
    # fallback: measure allocation for ψ.h(xk + sj + y)
    alloc1 = @allocated begin
      tmp = ψ.xk .+ ψ.sj .+ y
      val = ψ.h(tmp)
    end
    println("alloc for ψ.h(xk + sj + y) (tmp allocated): ", alloc1)
  else
    # simple case: no sj field, measure ψ.h(xk + y)
    alloc1 = @allocated begin
      tmp = ψ.xk .+ y
      val = ψ.h(tmp)
    end
    println("alloc for ψ.h(xk + y) (tmp allocated): ", alloc1)
  end

  # measure full ψ(y) call
  alloc_full = @allocated ψ(y)
  println("alloc for ψ(y): ", alloc_full)
end

inspect_shifted_eval(:NormL1)
inspect_shifted_eval(:NormL0)
inspect_shifted_eval(:RootNormLhalf)
