uint64_t helper_fre (uint64_t arg)

{

    CPU_DoubleU fone, farg;

    fone.ll = 0x3FF0000000000000ULL; /* 1.0 */

    farg.ll = arg;



    if (unlikely(float64_is_signaling_nan(farg.d))) {

        /* sNaN reciprocal */

        farg.ll = fload_invalid_op_excp(POWERPC_EXCP_FP_VXSNAN);

    } else if (unlikely(float64_is_zero(farg.d))) {

        /* Zero reciprocal */

        farg.ll = float_zero_divide_excp(fone.d, farg.d);

    } else if (likely(isnormal(farg.d))) {

        farg.d = float64_div(fone.d, farg.d, &env->fp_status);

    } else {

        if (farg.ll == 0x8000000000000000ULL) {

            farg.ll = 0xFFF0000000000000ULL;

        } else if (farg.ll == 0x0000000000000000ULL) {

            farg.ll = 0x7FF0000000000000ULL;

        } else if (float64_is_nan(farg.d)) {

            farg.ll = 0x7FF8000000000000ULL;

        } else if (float64_is_neg(farg.d)) {

            farg.ll = 0x8000000000000000ULL;

        } else {

            farg.ll = 0x0000000000000000ULL;

        }

    }

    return farg.d;

}
