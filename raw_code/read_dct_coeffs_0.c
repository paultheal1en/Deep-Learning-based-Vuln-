static int read_dct_coeffs(BitstreamContext *bc, int32_t block[64],

                           const uint8_t *scan,

                           const int32_t quant_matrices[16][64], int q)

{

    int coef_list[128];

    int mode_list[128];

    int i, t, bits, ccoef, mode;

    int list_start = 64, list_end = 64, list_pos;

    int coef_count = 0;

    int coef_idx[64];

    int quant_idx;

    const int32_t *quant;



    coef_list[list_end] = 4;  mode_list[list_end++] = 0;

    coef_list[list_end] = 24; mode_list[list_end++] = 0;

    coef_list[list_end] = 44; mode_list[list_end++] = 0;

    coef_list[list_end] = 1;  mode_list[list_end++] = 3;

    coef_list[list_end] = 2;  mode_list[list_end++] = 3;

    coef_list[list_end] = 3;  mode_list[list_end++] = 3;



    for (bits = bitstream_read(bc, 4) - 1; bits >= 0; bits--) {

        list_pos = list_start;

        while (list_pos < list_end) {

            if (!(mode_list[list_pos] | coef_list[list_pos]) || !bitstream_read_bit(bc)) {

                list_pos++;

                continue;

            }

            ccoef = coef_list[list_pos];

            mode  = mode_list[list_pos];

            switch (mode) {

            case 0:

                coef_list[list_pos] = ccoef + 4;

                mode_list[list_pos] = 1;

            case 2:

                if (mode == 2) {

                    coef_list[list_pos]   = 0;

                    mode_list[list_pos++] = 0;

                }

                for (i = 0; i < 4; i++, ccoef++) {

                    if (bitstream_read_bit(bc)) {

                        coef_list[--list_start] = ccoef;

                        mode_list[  list_start] = 3;

                    } else {

                        if (!bits) {

                            t = 1 - (bitstream_read_bit(bc) << 1);

                        } else {

                            t = bitstream_read(bc, bits) | 1 << bits;

                            t = bitstream_apply_sign(bc, t);

                        }

                        block[scan[ccoef]] = t;

                        coef_idx[coef_count++] = ccoef;

                    }

                }

                break;

            case 1:

                mode_list[list_pos] = 2;

                for (i = 0; i < 3; i++) {

                    ccoef += 4;

                    coef_list[list_end]   = ccoef;

                    mode_list[list_end++] = 2;

                }

                break;

            case 3:

                if (!bits) {

                    t = 1 - (bitstream_read_bit(bc) << 1);

                } else {

                    t = bitstream_read(bc, bits) | 1 << bits;

                    t = bitstream_apply_sign(bc, t);

                }

                block[scan[ccoef]] = t;

                coef_idx[coef_count++] = ccoef;

                coef_list[list_pos]   = 0;

                mode_list[list_pos++] = 0;

                break;

            }

        }

    }



    if (q == -1) {

        quant_idx = bitstream_read(bc, 4);

    } else {

        quant_idx = q;

    }



    if (quant_idx >= 16)

        return AVERROR_INVALIDDATA;



    quant = quant_matrices[quant_idx];



    block[0] = (block[0] * quant[0]) >> 11;

    for (i = 0; i < coef_count; i++) {

        int idx = coef_idx[i];

        block[scan[idx]] = (block[scan[idx]] * quant[idx]) >> 11;

    }



    return 0;

}
