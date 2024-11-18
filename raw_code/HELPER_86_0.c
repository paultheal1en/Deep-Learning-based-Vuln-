uint32_t HELPER(lpebr)(CPUS390XState *env, uint32_t f1, uint32_t f2)

{

    float32 v1;

    float32 v2 = env->fregs[f2].d;



    v1 = float32_abs(v2);

    env->fregs[f1].d = v1;

    return set_cc_nz_f32(v1);

}
