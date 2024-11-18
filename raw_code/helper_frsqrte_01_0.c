uint64_t helper_frsqrte (uint64_t arg)

{

    CPU_DoubleU fone, farg;

    fone.ll = 0x3FF0000000000000ULL; /* 1.0 */

    farg.ll = arg;



    if (unlikely(float64_is_signaling_nan(farg.d))) {

        /* sNaN reciprocal square root */

        farg.ll = fload_invalid_op_excp(POWERPC_EXCP_FP_VXSNAN);

    } else if (unlikely(float64_is_neg(farg.d) && !float64_is_zero(farg.d))) {

        /* Reciprocal square root of a negative nonzero number */

        farg.ll = fload_invalid_op_excp(POWERPC_EXCP_FP_VXSQRT);

    } else if (likely(isnormal(farg.d))) {

        farg.d = float64_sqrt(farg.d, &env->fp_status);

        farg.d = float32_div(fone.d, farg.d, &env->fp_status);

    } else {

        if (farg.ll == 0x8000000000000000ULL) {

            farg.ll = 0xFFF0000000000000ULL;

        } else if (farg.ll == 0x0000000000000000ULL) {

            farg.ll = 0x7FF0000000000000ULL;

        } else if (float64_is_nan(farg.d)) {

            farg.ll |= 0x000FFFFFFFFFFFFFULL;

        } else if (float64_is_neg(farg.d)) {

            farg.ll = 0x7FF8000000000000ULL;

        } else {

            farg.ll = 0x0000000000000000ULL;

        }

    }

    return farg.ll;

}
