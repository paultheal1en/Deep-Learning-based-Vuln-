uint32_t helper_compute_fprf (uint64_t arg, uint32_t set_fprf)

{

    CPU_DoubleU farg;

    int isneg;

    int ret;

    farg.ll = arg;

    isneg = float64_is_neg(farg.d);

    if (unlikely(float64_is_nan(farg.d))) {

        if (float64_is_signaling_nan(farg.d)) {

            /* Signaling NaN: flags are undefined */

            ret = 0x00;

        } else {

            /* Quiet NaN */

            ret = 0x11;

        }

    } else if (unlikely(float64_is_infinity(farg.d))) {

        /* +/- infinity */

        if (isneg)

            ret = 0x09;

        else

            ret = 0x05;

    } else {

        if (float64_is_zero(farg.d)) {

            /* +/- zero */

            if (isneg)

                ret = 0x12;

            else

                ret = 0x02;

        } else {

            if (isden(farg.d)) {

                /* Denormalized numbers */

                ret = 0x10;

            } else {

                /* Normalized numbers */

                ret = 0x00;

            }

            if (isneg) {

                ret |= 0x08;

            } else {

                ret |= 0x04;

            }

        }

    }

    if (set_fprf) {

        /* We update FPSCR_FPRF */

        env->fpscr &= ~(0x1F << FPSCR_FPRF);

        env->fpscr |= ret << FPSCR_FPRF;

    }

    /* We just need fpcc to update Rc1 */

    return ret & 0xF;

}
