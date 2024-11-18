int32_t helper_fdtoi(CPUSPARCState *env, float64 src)

{

    int32_t ret;

    clear_float_exceptions(env);

    ret = float64_to_int32_round_to_zero(src, &env->fp_status);

    check_ieee_exceptions(env);

    return ret;

}
