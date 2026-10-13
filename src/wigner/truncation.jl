"""
    calc_Mmax(cp, l) -> Int

Return the retained global magnetic-index limit at multipole l.
"""
@inline function calc_Mmax(cp, l::Integer)
    return min(cp.κ + cp.bnmax, Int(l))
end

"""
    calc_nmax(cp, l) -> Int

Return the retained beam-index limit at multipole l.
"""
@inline function calc_nmax(cp, l::Integer)
    return min(cp.bnmax, Int(l))
end
