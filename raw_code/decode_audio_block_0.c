static int decode_audio_block(AC3DecodeContext *s, int blk)

{

    int fbw_channels = s->fbw_channels;

    int channel_mode = s->channel_mode;

    int i, bnd, seg, ch;

    int different_transforms;

    int downmix_output;

    int cpl_in_use;

    GetBitContext *gbc = &s->gbc;

    uint8_t bit_alloc_stages[AC3_MAX_CHANNELS];



    memset(bit_alloc_stages, 0, AC3_MAX_CHANNELS);



    /* block switch flags */

    different_transforms = 0;

    if (s->block_switch_syntax) {

        for (ch = 1; ch <= fbw_channels; ch++) {

            s->block_switch[ch] = get_bits1(gbc);

            if(ch > 1 && s->block_switch[ch] != s->block_switch[1])

                different_transforms = 1;

        }

    }



    /* dithering flags */

    if (s->dither_flag_syntax) {

        for (ch = 1; ch <= fbw_channels; ch++) {

            s->dither_flag[ch] = get_bits1(gbc);

        }

    }



    /* dynamic range */

    i = !(s->channel_mode);

    do {

        if(get_bits1(gbc)) {

            s->dynamic_range[i] = ((dynamic_range_tab[get_bits(gbc, 8)]-1.0) *

                                  s->avctx->drc_scale)+1.0;

        } else if(blk == 0) {

            s->dynamic_range[i] = 1.0f;

        }

    } while(i--);



    /* spectral extension strategy */

    if (s->eac3 && (!blk || get_bits1(gbc))) {

        if (get_bits1(gbc)) {

            av_log_missing_feature(s->avctx, "Spectral extension", 1);

            return -1;

        }

        /* TODO: parse spectral extension strategy info */

    }



    /* TODO: spectral extension coordinates */



    /* coupling strategy */

    if (s->eac3 ? s->cpl_strategy_exists[blk] : get_bits1(gbc)) {

        memset(bit_alloc_stages, 3, AC3_MAX_CHANNELS);

        if (!s->eac3)

            s->cpl_in_use[blk] = get_bits1(gbc);

        if (s->cpl_in_use[blk]) {

            /* coupling in use */

            int cpl_start_subband, cpl_end_subband;



            if (channel_mode < AC3_CHMODE_STEREO) {

                av_log(s->avctx, AV_LOG_ERROR, "coupling not allowed in mono or dual-mono\n");

                return -1;

            }



            /* check for enhanced coupling */

            if (s->eac3 && get_bits1(gbc)) {

                /* TODO: parse enhanced coupling strategy info */

                av_log_missing_feature(s->avctx, "Enhanced coupling", 1);

                return -1;

            }



            /* determine which channels are coupled */

            if (s->eac3 && s->channel_mode == AC3_CHMODE_STEREO) {

                s->channel_in_cpl[1] = 1;

                s->channel_in_cpl[2] = 1;

            } else {

                for (ch = 1; ch <= fbw_channels; ch++)

                    s->channel_in_cpl[ch] = get_bits1(gbc);

            }



            /* phase flags in use */

            if (channel_mode == AC3_CHMODE_STEREO)

                s->phase_flags_in_use = get_bits1(gbc);



            /* coupling frequency range */

            /* TODO: modify coupling end freq if spectral extension is used */

            cpl_start_subband = get_bits(gbc, 4);

            cpl_end_subband   = get_bits(gbc, 4) + 3;

            s->num_cpl_subbands = cpl_end_subband - cpl_start_subband;

            if (s->num_cpl_subbands < 0) {

                av_log(s->avctx, AV_LOG_ERROR, "invalid coupling range (%d > %d)\n",

                       cpl_start_subband, cpl_end_subband);

                return -1;

            }

            s->start_freq[CPL_CH] = cpl_start_subband * 12 + 37;

            s->end_freq[CPL_CH]   = cpl_end_subband   * 12 + 37;



           decode_band_structure(gbc, blk, s->eac3, 0,

                                 cpl_start_subband, cpl_end_subband,

                                 ff_eac3_default_cpl_band_struct,

                                 s->cpl_band_struct, &s->num_cpl_subbands,

                                 &s->num_cpl_bands, NULL);

        } else {

            /* coupling not in use */

            for (ch = 1; ch <= fbw_channels; ch++) {

                s->channel_in_cpl[ch] = 0;

                s->first_cpl_coords[ch] = 1;

            }

            s->first_cpl_leak = s->eac3;

            s->phase_flags_in_use = 0;

        }

    } else if (!s->eac3) {

        if(!blk) {

            av_log(s->avctx, AV_LOG_ERROR, "new coupling strategy must be present in block 0\n");

            return -1;

        } else {

            s->cpl_in_use[blk] = s->cpl_in_use[blk-1];

        }

    }

    cpl_in_use = s->cpl_in_use[blk];



    /* coupling coordinates */

    if (cpl_in_use) {

        int cpl_coords_exist = 0;



        for (ch = 1; ch <= fbw_channels; ch++) {

            if (s->channel_in_cpl[ch]) {

                if ((s->eac3 && s->first_cpl_coords[ch]) || get_bits1(gbc)) {

                    int master_cpl_coord, cpl_coord_exp, cpl_coord_mant;

                    s->first_cpl_coords[ch] = 0;

                    cpl_coords_exist = 1;

                    master_cpl_coord = 3 * get_bits(gbc, 2);

                    for (bnd = 0; bnd < s->num_cpl_bands; bnd++) {

                        cpl_coord_exp = get_bits(gbc, 4);

                        cpl_coord_mant = get_bits(gbc, 4);

                        if (cpl_coord_exp == 15)

                            s->cpl_coords[ch][bnd] = cpl_coord_mant << 22;

                        else

                            s->cpl_coords[ch][bnd] = (cpl_coord_mant + 16) << 21;

                        s->cpl_coords[ch][bnd] >>= (cpl_coord_exp + master_cpl_coord);

                    }

                } else if (!blk) {

                    av_log(s->avctx, AV_LOG_ERROR, "new coupling coordinates must be present in block 0\n");

                    return -1;

                }

            } else {

                /* channel not in coupling */

                s->first_cpl_coords[ch] = 1;

            }

        }

        /* phase flags */

        if (channel_mode == AC3_CHMODE_STEREO && cpl_coords_exist) {

            for (bnd = 0; bnd < s->num_cpl_bands; bnd++) {

                s->phase_flags[bnd] = s->phase_flags_in_use? get_bits1(gbc) : 0;

            }

        }

    }



    /* stereo rematrixing strategy and band structure */

    if (channel_mode == AC3_CHMODE_STEREO) {

        if ((s->eac3 && !blk) || get_bits1(gbc)) {

            s->num_rematrixing_bands = 4;

            if(cpl_in_use && s->start_freq[CPL_CH] <= 61)

                s->num_rematrixing_bands -= 1 + (s->start_freq[CPL_CH] == 37);

            for(bnd=0; bnd<s->num_rematrixing_bands; bnd++)

                s->rematrixing_flags[bnd] = get_bits1(gbc);

        } else if (!blk) {

            av_log(s->avctx, AV_LOG_ERROR, "new rematrixing strategy must be present in block 0\n");

            return -1;

        }

    }



    /* exponent strategies for each channel */

    for (ch = !cpl_in_use; ch <= s->channels; ch++) {

        if (!s->eac3)

            s->exp_strategy[blk][ch] = get_bits(gbc, 2 - (ch == s->lfe_ch));

        if(s->exp_strategy[blk][ch] != EXP_REUSE)

            bit_alloc_stages[ch] = 3;

    }



    /* channel bandwidth */

    for (ch = 1; ch <= fbw_channels; ch++) {

        s->start_freq[ch] = 0;

        if (s->exp_strategy[blk][ch] != EXP_REUSE) {

            int group_size;

            int prev = s->end_freq[ch];

            if (s->channel_in_cpl[ch])

                s->end_freq[ch] = s->start_freq[CPL_CH];

            else {

                int bandwidth_code = get_bits(gbc, 6);

                if (bandwidth_code > 60) {

                    av_log(s->avctx, AV_LOG_ERROR, "bandwidth code = %d > 60\n", bandwidth_code);

                    return -1;

                }

                s->end_freq[ch] = bandwidth_code * 3 + 73;

            }

            group_size = 3 << (s->exp_strategy[blk][ch] - 1);

            s->num_exp_groups[ch] = (s->end_freq[ch]+group_size-4) / group_size;

            if(blk > 0 && s->end_freq[ch] != prev)

                memset(bit_alloc_stages, 3, AC3_MAX_CHANNELS);

        }

    }

    if (cpl_in_use && s->exp_strategy[blk][CPL_CH] != EXP_REUSE) {

        s->num_exp_groups[CPL_CH] = (s->end_freq[CPL_CH] - s->start_freq[CPL_CH]) /

                                    (3 << (s->exp_strategy[blk][CPL_CH] - 1));

    }



    /* decode exponents for each channel */

    for (ch = !cpl_in_use; ch <= s->channels; ch++) {

        if (s->exp_strategy[blk][ch] != EXP_REUSE) {

            s->dexps[ch][0] = get_bits(gbc, 4) << !ch;

            if (decode_exponents(gbc, s->exp_strategy[blk][ch],

                                 s->num_exp_groups[ch], s->dexps[ch][0],

                                 &s->dexps[ch][s->start_freq[ch]+!!ch])) {

                av_log(s->avctx, AV_LOG_ERROR, "exponent out-of-range\n");

                return -1;

            }

            if(ch != CPL_CH && ch != s->lfe_ch)

                skip_bits(gbc, 2); /* skip gainrng */

        }

    }



    /* bit allocation information */

    if (s->bit_allocation_syntax) {

        if (get_bits1(gbc)) {

            s->bit_alloc_params.slow_decay = ff_ac3_slow_decay_tab[get_bits(gbc, 2)] >> s->bit_alloc_params.sr_shift;

            s->bit_alloc_params.fast_decay = ff_ac3_fast_decay_tab[get_bits(gbc, 2)] >> s->bit_alloc_params.sr_shift;

            s->bit_alloc_params.slow_gain  = ff_ac3_slow_gain_tab[get_bits(gbc, 2)];

            s->bit_alloc_params.db_per_bit = ff_ac3_db_per_bit_tab[get_bits(gbc, 2)];

            s->bit_alloc_params.floor  = ff_ac3_floor_tab[get_bits(gbc, 3)];

            for(ch=!cpl_in_use; ch<=s->channels; ch++)

                bit_alloc_stages[ch] = FFMAX(bit_alloc_stages[ch], 2);

        } else if (!blk) {

            av_log(s->avctx, AV_LOG_ERROR, "new bit allocation info must be present in block 0\n");

            return -1;

        }

    }



    /* signal-to-noise ratio offsets and fast gains (signal-to-mask ratios) */

    if(!s->eac3 || !blk){

        if(s->snr_offset_strategy && get_bits1(gbc)) {

            int snr = 0;

            int csnr;

            csnr = (get_bits(gbc, 6) - 15) << 4;

            for (i = ch = !cpl_in_use; ch <= s->channels; ch++) {

                /* snr offset */

                if (ch == i || s->snr_offset_strategy == 2)

                    snr = (csnr + get_bits(gbc, 4)) << 2;

                /* run at least last bit allocation stage if snr offset changes */

                if(blk && s->snr_offset[ch] != snr) {

                    bit_alloc_stages[ch] = FFMAX(bit_alloc_stages[ch], 1);

                }

                s->snr_offset[ch] = snr;



                /* fast gain (normal AC-3 only) */

                if (!s->eac3) {

                    int prev = s->fast_gain[ch];

                    s->fast_gain[ch] = ff_ac3_fast_gain_tab[get_bits(gbc, 3)];

                    /* run last 2 bit allocation stages if fast gain changes */

                    if(blk && prev != s->fast_gain[ch])

                        bit_alloc_stages[ch] = FFMAX(bit_alloc_stages[ch], 2);

                }

            }

        } else if (!s->eac3 && !blk) {

            av_log(s->avctx, AV_LOG_ERROR, "new snr offsets must be present in block 0\n");

            return -1;

        }

    }



    /* fast gain (E-AC-3 only) */

    if (s->fast_gain_syntax && get_bits1(gbc)) {

        for (ch = !cpl_in_use; ch <= s->channels; ch++) {

            int prev = s->fast_gain[ch];

            s->fast_gain[ch] = ff_ac3_fast_gain_tab[get_bits(gbc, 3)];

            /* run last 2 bit allocation stages if fast gain changes */

            if(blk && prev != s->fast_gain[ch])

                bit_alloc_stages[ch] = FFMAX(bit_alloc_stages[ch], 2);

        }

    } else if (s->eac3 && !blk) {

        for (ch = !cpl_in_use; ch <= s->channels; ch++)

            s->fast_gain[ch] = ff_ac3_fast_gain_tab[4];

    }



    /* E-AC-3 to AC-3 converter SNR offset */

    if (s->frame_type == EAC3_FRAME_TYPE_INDEPENDENT && get_bits1(gbc)) {

        skip_bits(gbc, 10); // skip converter snr offset

    }



    /* coupling leak information */

    if (cpl_in_use) {

        if (s->first_cpl_leak || get_bits1(gbc)) {

            int fl = get_bits(gbc, 3);

            int sl = get_bits(gbc, 3);

            /* run last 2 bit allocation stages for coupling channel if

               coupling leak changes */

            if(blk && (fl != s->bit_alloc_params.cpl_fast_leak ||

                       sl != s->bit_alloc_params.cpl_slow_leak)) {

                bit_alloc_stages[CPL_CH] = FFMAX(bit_alloc_stages[CPL_CH], 2);

            }

            s->bit_alloc_params.cpl_fast_leak = fl;

            s->bit_alloc_params.cpl_slow_leak = sl;

        } else if (!s->eac3 && !blk) {

            av_log(s->avctx, AV_LOG_ERROR, "new coupling leak info must be present in block 0\n");

            return -1;

        }

        s->first_cpl_leak = 0;

    }



    /* delta bit allocation information */

    if (s->dba_syntax && get_bits1(gbc)) {

        /* delta bit allocation exists (strategy) */

        for (ch = !cpl_in_use; ch <= fbw_channels; ch++) {

            s->dba_mode[ch] = get_bits(gbc, 2);

            if (s->dba_mode[ch] == DBA_RESERVED) {

                av_log(s->avctx, AV_LOG_ERROR, "delta bit allocation strategy reserved\n");

                return -1;

            }

            bit_alloc_stages[ch] = FFMAX(bit_alloc_stages[ch], 2);

        }

        /* channel delta offset, len and bit allocation */

        for (ch = !cpl_in_use; ch <= fbw_channels; ch++) {

            if (s->dba_mode[ch] == DBA_NEW) {

                s->dba_nsegs[ch] = get_bits(gbc, 3);

                for (seg = 0; seg <= s->dba_nsegs[ch]; seg++) {

                    s->dba_offsets[ch][seg] = get_bits(gbc, 5);

                    s->dba_lengths[ch][seg] = get_bits(gbc, 4);

                    s->dba_values[ch][seg] = get_bits(gbc, 3);

                }

                /* run last 2 bit allocation stages if new dba values */

                bit_alloc_stages[ch] = FFMAX(bit_alloc_stages[ch], 2);

            }

        }

    } else if(blk == 0) {

        for(ch=0; ch<=s->channels; ch++) {

            s->dba_mode[ch] = DBA_NONE;

        }

    }



    /* Bit allocation */

    for(ch=!cpl_in_use; ch<=s->channels; ch++) {

        if(bit_alloc_stages[ch] > 2) {

            /* Exponent mapping into PSD and PSD integration */

            ff_ac3_bit_alloc_calc_psd(s->dexps[ch],

                                      s->start_freq[ch], s->end_freq[ch],

                                      s->psd[ch], s->band_psd[ch]);

        }

        if(bit_alloc_stages[ch] > 1) {

            /* Compute excitation function, Compute masking curve, and

               Apply delta bit allocation */

            ff_ac3_bit_alloc_calc_mask(&s->bit_alloc_params, s->band_psd[ch],

                                       s->start_freq[ch], s->end_freq[ch],

                                       s->fast_gain[ch], (ch == s->lfe_ch),

                                       s->dba_mode[ch], s->dba_nsegs[ch],

                                       s->dba_offsets[ch], s->dba_lengths[ch],

                                       s->dba_values[ch], s->mask[ch]);

        }

        if(bit_alloc_stages[ch] > 0) {

            /* Compute bit allocation */

            const uint8_t *bap_tab = s->channel_uses_aht[ch] ?

                                     ff_eac3_hebap_tab : ff_ac3_bap_tab;

            ff_ac3_bit_alloc_calc_bap(s->mask[ch], s->psd[ch],

                                      s->start_freq[ch], s->end_freq[ch],

                                      s->snr_offset[ch],

                                      s->bit_alloc_params.floor,

                                      bap_tab, s->bap[ch]);

        }

    }



    /* unused dummy data */

    if (s->skip_syntax && get_bits1(gbc)) {

        int skipl = get_bits(gbc, 9);

        while(skipl--)

            skip_bits(gbc, 8);

    }



    /* unpack the transform coefficients

       this also uncouples channels if coupling is in use. */

    decode_transform_coeffs(s, blk);



    /* TODO: generate enhanced coupling coordinates and uncouple */



    /* TODO: apply spectral extension */



    /* recover coefficients if rematrixing is in use */

    if(s->channel_mode == AC3_CHMODE_STEREO)

        do_rematrixing(s);



    /* apply scaling to coefficients (headroom, dynrng) */

    for(ch=1; ch<=s->channels; ch++) {

        float gain = s->mul_bias / 4194304.0f;

        if(s->channel_mode == AC3_CHMODE_DUALMONO) {

            gain *= s->dynamic_range[ch-1];

        } else {

            gain *= s->dynamic_range[0];

        }

        s->dsp.int32_to_float_fmul_scalar(s->transform_coeffs[ch], s->fixed_coeffs[ch], gain, 256);

    }



    /* downmix and MDCT. order depends on whether block switching is used for

       any channel in this block. this is because coefficients for the long

       and short transforms cannot be mixed. */

    downmix_output = s->channels != s->out_channels &&

                     !((s->output_mode & AC3_OUTPUT_LFEON) &&

                     s->fbw_channels == s->out_channels);

    if(different_transforms) {

        /* the delay samples have already been downmixed, so we upmix the delay

           samples in order to reconstruct all channels before downmixing. */

        if(s->downmixed) {

            s->downmixed = 0;

            ac3_upmix_delay(s);

        }



        do_imdct(s, s->channels);



        if(downmix_output) {

            s->dsp.ac3_downmix(s->output, s->downmix_coeffs, s->out_channels, s->fbw_channels, 256);

        }

    } else {

        if(downmix_output) {

            s->dsp.ac3_downmix(s->transform_coeffs+1, s->downmix_coeffs, s->out_channels, s->fbw_channels, 256);

        }



        if(downmix_output && !s->downmixed) {

            s->downmixed = 1;

            s->dsp.ac3_downmix(s->delay, s->downmix_coeffs, s->out_channels, s->fbw_channels, 128);

        }



        do_imdct(s, s->out_channels);

    }



    return 0;

}
