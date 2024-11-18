void helper_ctc1(CPUMIPSState *env, target_ulong arg1, uint32_t reg)

{

    switch(reg) {

    case 25:

        if (arg1 & 0xffffff00)

            return;

        env->active_fpu.fcr31 = (env->active_fpu.fcr31 & 0x017fffff) | ((arg1 & 0xfe) << 24) |

                     ((arg1 & 0x1) << 23);

        break;

    case 26:

        if (arg1 & 0x007c0000)

            return;

        env->active_fpu.fcr31 = (env->active_fpu.fcr31 & 0xfffc0f83) | (arg1 & 0x0003f07c);

        break;

    case 28:

        if (arg1 & 0x007c0000)

            return;

        env->active_fpu.fcr31 = (env->active_fpu.fcr31 & 0xfefff07c) | (arg1 & 0x00000f83) |

                     ((arg1 & 0x4) << 22);

        break;

    case 31:

        if (arg1 & 0x007c0000)

            return;

        env->active_fpu.fcr31 = arg1;

        break;

    default:

        return;

    }

    /* set rounding mode */

    restore_rounding_mode(env);

    /* set flush-to-zero mode */

    restore_flush_mode(env);

    set_float_exception_flags(0, &env->active_fpu.fp_status);

    if ((GET_FP_ENABLE(env->active_fpu.fcr31) | 0x20) & GET_FP_CAUSE(env->active_fpu.fcr31))

        do_raise_exception(env, EXCP_FPE, GETPC());

}
