void helper_ldl_l_raw(uint64_t t0, uint64_t t1)

{

    env->lock = t1;

    ldl_raw(t1, t0);

}
