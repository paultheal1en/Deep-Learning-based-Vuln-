uint32_t HELPER(lpdbr)(CPUS390XState *env, uint32_t f1, uint32_t f2)

{

    float64 v1;

    float64 v2 = env->fregs[f2].d;



    v1 = float64_abs(v2);

    env->fregs[f1].d = v1;

    return set_cc_nz_f64(v1);

}
