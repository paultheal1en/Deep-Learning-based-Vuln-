static void synth_block_fcb_acb(WMAVoiceContext *s, GetBitContext *gb,

                                int block_idx, int size,

                                int block_pitch_sh2,

                                const struct frame_type_desc *frame_desc,

                                float *excitation)

{

    static const float gain_coeff[6] = {

        0.8169, -0.06545, 0.1726, 0.0185, -0.0359, 0.0458

    };

    float pulses[MAX_FRAMESIZE / 2], pred_err, acb_gain, fcb_gain;

    int n, idx, gain_weight;

    AMRFixed fcb;



    assert(size <= MAX_FRAMESIZE / 2);

    memset(pulses, 0, sizeof(*pulses) * size);



    fcb.pitch_lag      = block_pitch_sh2 >> 2;

    fcb.pitch_fac      = 1.0;

    fcb.no_repeat_mask = 0;

    fcb.n              = 0;



    /* For the other frame types, this is where we apply the innovation

     * (fixed) codebook pulses of the speech signal. */

    if (frame_desc->fcb_type == FCB_TYPE_AW_PULSES) {

        aw_pulse_set1(s, gb, block_idx, &fcb);

        aw_pulse_set2(s, gb, block_idx, &fcb);

    } else /* FCB_TYPE_EXC_PULSES */ {

        int offset_nbits = 5 - frame_desc->log_n_blocks;



        fcb.no_repeat_mask = -1;

        /* similar to ff_decode_10_pulses_35bits(), but with single pulses

         * (instead of double) for a subset of pulses */

        for (n = 0; n < 5; n++) {

            float sign;

            int pos1, pos2;



            sign           = get_bits1(gb) ? 1.0 : -1.0;

            pos1           = get_bits(gb, offset_nbits);

            fcb.x[fcb.n]   = n + 5 * pos1;

            fcb.y[fcb.n++] = sign;

            if (n < frame_desc->dbl_pulses) {

                pos2           = get_bits(gb, offset_nbits);

                fcb.x[fcb.n]   = n + 5 * pos2;

                fcb.y[fcb.n++] = (pos1 < pos2) ? -sign : sign;

            }

        }

    }

    ff_set_fixed_vector(pulses, &fcb, 1.0, size);



    /* Calculate gain for adaptive & fixed codebook signal.

     * see ff_amr_set_fixed_gain(). */

    idx = get_bits(gb, 7);

    fcb_gain = expf(avpriv_scalarproduct_float_c(s->gain_pred_err,

                                                 gain_coeff, 6) -

                    5.2409161640 + wmavoice_gain_codebook_fcb[idx]);

    acb_gain = wmavoice_gain_codebook_acb[idx];

    pred_err = av_clipf(wmavoice_gain_codebook_fcb[idx],

                        -2.9957322736 /* log(0.05) */,

                         1.6094379124 /* log(5.0)  */);



    gain_weight = 8 >> frame_desc->log_n_blocks;

    memmove(&s->gain_pred_err[gain_weight], s->gain_pred_err,

            sizeof(*s->gain_pred_err) * (6 - gain_weight));

    for (n = 0; n < gain_weight; n++)

        s->gain_pred_err[n] = pred_err;



    /* Calculation of adaptive codebook */

    if (frame_desc->acb_type == ACB_TYPE_ASYMMETRIC) {

        int len;

        for (n = 0; n < size; n += len) {

            int next_idx_sh16;

            int abs_idx    = block_idx * size + n;

            int pitch_sh16 = (s->last_pitch_val << 16) +

                             s->pitch_diff_sh16 * abs_idx;

            int pitch      = (pitch_sh16 + 0x6FFF) >> 16;

            int idx_sh16   = ((pitch << 16) - pitch_sh16) * 8 + 0x58000;

            idx            = idx_sh16 >> 16;

            if (s->pitch_diff_sh16) {

                if (s->pitch_diff_sh16 > 0) {

                    next_idx_sh16 = (idx_sh16) &~ 0xFFFF;

                } else

                    next_idx_sh16 = (idx_sh16 + 0x10000) &~ 0xFFFF;

                len = av_clip((idx_sh16 - next_idx_sh16) / s->pitch_diff_sh16 / 8,

                              1, size - n);

            } else

                len = size;



            ff_acelp_interpolatef(&excitation[n], &excitation[n - pitch],

                                  wmavoice_ipol1_coeffs, 17,

                                  idx, 9, len);

        }

    } else /* ACB_TYPE_HAMMING */ {

        int block_pitch = block_pitch_sh2 >> 2;

        idx             = block_pitch_sh2 & 3;

        if (idx) {

            ff_acelp_interpolatef(excitation, &excitation[-block_pitch],

                                  wmavoice_ipol2_coeffs, 4,

                                  idx, 8, size);

        } else

            av_memcpy_backptr((uint8_t *) excitation, sizeof(float) * block_pitch,

                              sizeof(float) * size);

    }



    /* Interpolate ACB/FCB and use as excitation signal */

    ff_weighted_vector_sumf(excitation, excitation, pulses,

                            acb_gain, fcb_gain, size);

}
