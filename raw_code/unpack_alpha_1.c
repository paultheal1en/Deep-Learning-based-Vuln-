static void unpack_alpha(GetBitContext *gb, uint16_t *dst, int num_coeffs,

                         const int num_bits)

{

    const int mask = (1 << num_bits) - 1;

    int i, idx, val, alpha_val;



    idx       = 0;

    alpha_val = mask;

    do {

        do {

            if (get_bits1(gb))

                val = get_bits(gb, num_bits);

            else {

                int sign;

                val  = get_bits(gb, num_bits == 16 ? 7 : 4);

                sign = val & 1;

                val  = (val + 2) >> 1;

                if (sign)

                    val = -val;

            }

            alpha_val = (alpha_val + val) & mask;

            if (num_bits == 16)

                dst[idx++] = alpha_val >> 6;

            else

                dst[idx++] = (alpha_val << 2) | (alpha_val >> 6);

            if (idx == num_coeffs - 1)

                break;

        } while (get_bits1(gb));

        val = get_bits(gb, 4);

        if (!val)

            val = get_bits(gb, 11);

        if (idx + val > num_coeffs)

            val = num_coeffs - idx;

        if (num_bits == 16)

            for (i = 0; i < val; i++)

                dst[idx++] = alpha_val >> 6;

        else

            for (i = 0; i < val; i++)

                dst[idx++] = (alpha_val << 2) | (alpha_val >> 6);

    } while (idx < num_coeffs);

}
