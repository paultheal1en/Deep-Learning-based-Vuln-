uint32_t HELPER(lcebr)(CPUS390XState *env, uint32_t f1, uint32_t f2)

{

    env->fregs[f1].l.upper = float32_chs(env->fregs[f2].l.upper);



    return set_cc_nz_f32(env->fregs[f1].l.upper);

}
