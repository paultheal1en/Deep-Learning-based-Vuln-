static inline int decode_ac_coeffs(GetBitContext *gb, int16_t *out,

                                   int blocks_per_slice,

                                   int plane_size_factor,

                                   const uint8_t *scan)

{

    int pos, block_mask, run, level, sign, run_cb_index, lev_cb_index;

    int max_coeffs, bits_left;



    /* set initial prediction values */

    run   = 4;

    level = 2;



    max_coeffs = blocks_per_slice << 6;

    block_mask = blocks_per_slice - 1;



    for (pos = blocks_per_slice - 1; pos < max_coeffs;) {

        run_cb_index = ff_prores_run_to_cb_index[FFMIN(run, 15)];

        lev_cb_index = ff_prores_lev_to_cb_index[FFMIN(level, 9)];



        bits_left = get_bits_left(gb);

        if (bits_left <= 0 || (bits_left <= 8 && !show_bits(gb, bits_left)))

            return 0;



        run = decode_vlc_codeword(gb, ff_prores_ac_codebook[run_cb_index]);

        if (run < 0)

            return AVERROR_INVALIDDATA;



        bits_left = get_bits_left(gb);

        if (bits_left <= 0 || (bits_left <= 8 && !show_bits(gb, bits_left)))

            return AVERROR_INVALIDDATA;



        level = decode_vlc_codeword(gb, ff_prores_ac_codebook[lev_cb_index]) + 1;

        if (level < 0)

            return AVERROR_INVALIDDATA;



        pos += run + 1;

        if (pos >= max_coeffs)

            break;



        sign = get_sbits(gb, 1);

        out[((pos & block_mask) << 6) + scan[pos >> plane_size_factor]] =

            (level ^ sign) - sign;

    }



    return 0;

}
