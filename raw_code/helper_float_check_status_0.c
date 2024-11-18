void helper_float_check_status (void)

{

#ifdef CONFIG_SOFTFLOAT

    if (env->exception_index == POWERPC_EXCP_PROGRAM &&

        (env->error_code & POWERPC_EXCP_FP)) {

        /* Differred floating-point exception after target FPR update */

        if (msr_fe0 != 0 || msr_fe1 != 0)

            helper_raise_exception_err(env->exception_index, env->error_code);

    } else {

        int status = get_float_exception_flags(&env->fp_status);

        if (status & float_flag_overflow) {

            float_overflow_excp();

        } else if (status & float_flag_underflow) {

            float_underflow_excp();

        } else if (status & float_flag_inexact) {

            float_inexact_excp();

        }

    }

#else

    if (env->exception_index == POWERPC_EXCP_PROGRAM &&

        (env->error_code & POWERPC_EXCP_FP)) {

        /* Differred floating-point exception after target FPR update */

        if (msr_fe0 != 0 || msr_fe1 != 0)

            helper_raise_exception_err(env->exception_index, env->error_code);

    }

#endif

}
