static inline int vfp_exceptbits_from_host(int host_bits)

{

    int target_bits = 0;



    if (host_bits & float_flag_invalid)

        target_bits |= 1;

    if (host_bits & float_flag_divbyzero)

        target_bits |= 2;

    if (host_bits & float_flag_overflow)

        target_bits |= 4;

    if (host_bits & float_flag_underflow)

        target_bits |= 8;

    if (host_bits & float_flag_inexact)

        target_bits |= 0x10;

    if (host_bits & float_flag_input_denormal)

        target_bits |= 0x80;

    return target_bits;

}
