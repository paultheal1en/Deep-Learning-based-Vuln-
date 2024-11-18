static av_cold void set_bandwidth(AC3EncodeContext *s)

{

    int blk, ch, cpl_start;



    if (s->cutoff) {

        /* calculate bandwidth based on user-specified cutoff frequency */

        int fbw_coeffs;

        fbw_coeffs     = s->cutoff * 2 * AC3_MAX_COEFS / s->sample_rate;

        s->bandwidth_code = av_clip((fbw_coeffs - 73) / 3, 0, 60);

    } else {

        /* use default bandwidth setting */

        s->bandwidth_code = ac3_bandwidth_tab[s->fbw_channels-1][s->bit_alloc.sr_code][s->frame_size_code/2];

    }



    /* set number of coefficients for each channel */

    for (ch = 1; ch <= s->fbw_channels; ch++) {

        s->start_freq[ch] = 0;

        for (blk = 0; blk < s->num_blocks; blk++)

            s->blocks[blk].end_freq[ch] = s->bandwidth_code * 3 + 73;

    }

    /* LFE channel always has 7 coefs */

    if (s->lfe_on) {

        s->start_freq[s->lfe_channel] = 0;

        for (blk = 0; blk < s->num_blocks; blk++)

            s->blocks[blk].end_freq[ch] = 7;

    }



    /* initialize coupling strategy */

    if (s->cpl_enabled) {

        if (s->options.cpl_start != AC3ENC_OPT_AUTO) {

            cpl_start = s->options.cpl_start;

        } else {

            cpl_start = ac3_coupling_start_tab[s->channel_mode-2][s->bit_alloc.sr_code][s->frame_size_code/2];

            if (cpl_start < 0) {

                if (s->options.channel_coupling == AC3ENC_OPT_AUTO)

                    s->cpl_enabled = 0;

                else

                    cpl_start = 15;

            }

        }

    }

    if (s->cpl_enabled) {

        int i, cpl_start_band, cpl_end_band;

        uint8_t *cpl_band_sizes = s->cpl_band_sizes;



        cpl_end_band   = s->bandwidth_code / 4 + 3;

        cpl_start_band = av_clip(cpl_start, 0, FFMIN(cpl_end_band-1, 15));



        s->num_cpl_subbands = cpl_end_band - cpl_start_band;



        s->num_cpl_bands = 1;

        *cpl_band_sizes  = 12;

        for (i = cpl_start_band + 1; i < cpl_end_band; i++) {

            if (ff_eac3_default_cpl_band_struct[i]) {

                *cpl_band_sizes += 12;

            } else {

                s->num_cpl_bands++;

                cpl_band_sizes++;

                *cpl_band_sizes = 12;

            }

        }



        s->start_freq[CPL_CH] = cpl_start_band * 12 + 37;

        s->cpl_end_freq       = cpl_end_band   * 12 + 37;

        for (blk = 0; blk < s->num_blocks; blk++)

            s->blocks[blk].end_freq[CPL_CH] = s->cpl_end_freq;

    }

}
