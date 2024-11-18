uint32_t HELPER(lpxbr)(CPUS390XState *env, uint32_t f1, uint32_t f2)

{

    CPU_QuadU v1;

    CPU_QuadU v2;



    v2.ll.upper = env->fregs[f2].ll;

    v2.ll.lower = env->fregs[f2 + 2].ll;

    v1.q = float128_abs(v2.q);

    env->fregs[f1].ll = v1.ll.upper;

    env->fregs[f1 + 2].ll = v1.ll.lower;

    return set_cc_nz_f128(v1.q);

}
