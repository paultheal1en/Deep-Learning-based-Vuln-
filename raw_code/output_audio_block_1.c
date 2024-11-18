static void output_audio_block(AC3EncodeContext *s, int blk)

{

    int ch, i, baie, bnd, got_cpl, ch0;

    AC3Block *block = &s->blocks[blk];



    /* block switching */

    if (!s->eac3) {

        for (ch = 0; ch < s->fbw_channels; ch++)

            put_bits(&s->pb, 1, 0);

    }



    /* dither flags */

    if (!s->eac3) {

        for (ch = 0; ch < s->fbw_channels; ch++)

            put_bits(&s->pb, 1, 1);

    }



    /* dynamic range codes */

    put_bits(&s->pb, 1, 0);



    /* spectral extension */

    if (s->eac3)

        put_bits(&s->pb, 1, 0);



    /* channel coupling */

    if (!s->eac3)

        put_bits(&s->pb, 1, block->new_cpl_strategy);

    if (block->new_cpl_strategy) {

        if (!s->eac3)

            put_bits(&s->pb, 1, block->cpl_in_use);

        if (block->cpl_in_use) {

            int start_sub, end_sub;

            if (s->eac3)

                put_bits(&s->pb, 1, 0); /* enhanced coupling */

            if (!s->eac3 || s->channel_mode != AC3_CHMODE_STEREO) {

                for (ch = 1; ch <= s->fbw_channels; ch++)

                    put_bits(&s->pb, 1, block->channel_in_cpl[ch]);

            }

            if (s->channel_mode == AC3_CHMODE_STEREO)

                put_bits(&s->pb, 1, 0); /* phase flags in use */

            start_sub = (s->start_freq[CPL_CH] - 37) / 12;

            end_sub   = (s->cpl_end_freq       - 37) / 12;

            put_bits(&s->pb, 4, start_sub);

            put_bits(&s->pb, 4, end_sub - 3);

            /* coupling band structure */

            if (s->eac3) {

                put_bits(&s->pb, 1, 0); /* use default */

            } else {

                for (bnd = start_sub+1; bnd < end_sub; bnd++)

                    put_bits(&s->pb, 1, ff_eac3_default_cpl_band_struct[bnd]);

            }

        }

    }



    /* coupling coordinates */

    if (block->cpl_in_use) {

        for (ch = 1; ch <= s->fbw_channels; ch++) {

            if (block->channel_in_cpl[ch]) {

                if (!s->eac3 || block->new_cpl_coords[ch] != 2)

                    put_bits(&s->pb, 1, block->new_cpl_coords[ch]);

                if (block->new_cpl_coords[ch]) {

                    put_bits(&s->pb, 2, block->cpl_master_exp[ch]);

                    for (bnd = 0; bnd < s->num_cpl_bands; bnd++) {

                        put_bits(&s->pb, 4, block->cpl_coord_exp [ch][bnd]);

                        put_bits(&s->pb, 4, block->cpl_coord_mant[ch][bnd]);

                    }

                }

            }

        }

    }



    /* stereo rematrixing */

    if (s->channel_mode == AC3_CHMODE_STEREO) {

        if (!s->eac3 || blk > 0)

            put_bits(&s->pb, 1, block->new_rematrixing_strategy);

        if (block->new_rematrixing_strategy) {

            /* rematrixing flags */

            for (bnd = 0; bnd < block->num_rematrixing_bands; bnd++)

                put_bits(&s->pb, 1, block->rematrixing_flags[bnd]);

        }

    }



    /* exponent strategy */

    if (!s->eac3) {

        for (ch = !block->cpl_in_use; ch <= s->fbw_channels; ch++)

            put_bits(&s->pb, 2, s->exp_strategy[ch][blk]);

        if (s->lfe_on)

            put_bits(&s->pb, 1, s->exp_strategy[s->lfe_channel][blk]);

    }



    /* bandwidth */

    for (ch = 1; ch <= s->fbw_channels; ch++) {

        if (s->exp_strategy[ch][blk] != EXP_REUSE && !block->channel_in_cpl[ch])

            put_bits(&s->pb, 6, s->bandwidth_code);

    }



    /* exponents */

    for (ch = !block->cpl_in_use; ch <= s->channels; ch++) {

        int nb_groups;

        int cpl = (ch == CPL_CH);



        if (s->exp_strategy[ch][blk] == EXP_REUSE)

            continue;



        /* DC exponent */

        put_bits(&s->pb, 4, block->grouped_exp[ch][0] >> cpl);



        /* exponent groups */

        nb_groups = exponent_group_tab[cpl][s->exp_strategy[ch][blk]-1][block->end_freq[ch]-s->start_freq[ch]];

        for (i = 1; i <= nb_groups; i++)

            put_bits(&s->pb, 7, block->grouped_exp[ch][i]);



        /* gain range info */

        if (ch != s->lfe_channel && !cpl)

            put_bits(&s->pb, 2, 0);

    }



    /* bit allocation info */

    if (!s->eac3) {

        baie = (blk == 0);

        put_bits(&s->pb, 1, baie);

        if (baie) {

            put_bits(&s->pb, 2, s->slow_decay_code);

            put_bits(&s->pb, 2, s->fast_decay_code);

            put_bits(&s->pb, 2, s->slow_gain_code);

            put_bits(&s->pb, 2, s->db_per_bit_code);

            put_bits(&s->pb, 3, s->floor_code);

        }

    }



    /* snr offset */

    if (!s->eac3) {

        put_bits(&s->pb, 1, block->new_snr_offsets);

        if (block->new_snr_offsets) {

            put_bits(&s->pb, 6, s->coarse_snr_offset);

            for (ch = !block->cpl_in_use; ch <= s->channels; ch++) {

                put_bits(&s->pb, 4, s->fine_snr_offset[ch]);

                put_bits(&s->pb, 3, s->fast_gain_code[ch]);

            }

        }

    } else {

        put_bits(&s->pb, 1, 0); /* no converter snr offset */

    }



    /* coupling leak */

    if (block->cpl_in_use) {

        if (!s->eac3 || block->new_cpl_leak != 2)

            put_bits(&s->pb, 1, block->new_cpl_leak);

        if (block->new_cpl_leak) {

            put_bits(&s->pb, 3, s->bit_alloc.cpl_fast_leak);

            put_bits(&s->pb, 3, s->bit_alloc.cpl_slow_leak);

        }

    }



    if (!s->eac3) {

        put_bits(&s->pb, 1, 0); /* no delta bit allocation */

        put_bits(&s->pb, 1, 0); /* no data to skip */

    }



    /* mantissas */

    got_cpl = !block->cpl_in_use;

    for (ch = 1; ch <= s->channels; ch++) {

        int b, q;



        if (!got_cpl && ch > 1 && block->channel_in_cpl[ch-1]) {

            ch0     = ch - 1;

            ch      = CPL_CH;

            got_cpl = 1;

        }

        for (i = s->start_freq[ch]; i < block->end_freq[ch]; i++) {

            q = block->qmant[ch][i];

            b = s->ref_bap[ch][blk][i];

            switch (b) {

            case 0:                                          break;

            case 1: if (q != 128) put_bits (&s->pb,   5, q); break;

            case 2: if (q != 128) put_bits (&s->pb,   7, q); break;

            case 3:               put_sbits(&s->pb,   3, q); break;

            case 4: if (q != 128) put_bits (&s->pb,   7, q); break;

            case 14:              put_sbits(&s->pb,  14, q); break;

            case 15:              put_sbits(&s->pb,  16, q); break;

            default:              put_sbits(&s->pb, b-1, q); break;

            }

        }

        if (ch == CPL_CH)

            ch = ch0;

    }

}
