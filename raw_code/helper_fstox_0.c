int64_t helper_fstox(CPUSPARCState *env, float32 src)

{

    int64_t ret;

    clear_float_exceptions(env);

    ret = float32_to_int64_round_to_zero(src, &env->fp_status);

    check_ieee_exceptions(env);

    return ret;

}
