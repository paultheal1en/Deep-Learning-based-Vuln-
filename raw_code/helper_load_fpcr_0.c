uint64_t helper_load_fpcr (void)

{

    uint64_t ret = 0;

#ifdef CONFIG_SOFTFLOAT

    ret |= env->fp_status.float_exception_flags << 52;

    if (env->fp_status.float_exception_flags)

        ret |= 1ULL << 63;

    env->ipr[IPR_EXC_SUM] &= ~0x3E:

    env->ipr[IPR_EXC_SUM] |= env->fp_status.float_exception_flags << 1;

#endif

    switch (env->fp_status.float_rounding_mode) {

    case float_round_nearest_even:

        ret |= 2ULL << 58;

        break;

    case float_round_down:

        ret |= 1ULL << 58;

        break;

    case float_round_up:

        ret |= 3ULL << 58;

        break;

    case float_round_to_zero:

        break;

    }

    return ret;

}
