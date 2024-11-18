static int dca_subsubframe(DCAContext *s, int base_channel, int block_index)

{

    int k, l;

    int subsubframe = s->current_subsubframe;



    const float *quant_step_table;



    /* FIXME */

    float (*subband_samples)[DCA_SUBBANDS][8] = s->subband_samples[block_index];

    LOCAL_ALIGNED_16(int32_t, block, [8 * DCA_SUBBANDS]);



    /*

     * Audio data

     */



    /* Select quantization step size table */

    if (s->bit_rate_index == 0x1f)

        quant_step_table = lossless_quant_d;

    else

        quant_step_table = lossy_quant_d;



    for (k = base_channel; k < s->prim_channels; k++) {

        float rscale[DCA_SUBBANDS];



        if (get_bits_left(&s->gb) < 0)

            return AVERROR_INVALIDDATA;



        for (l = 0; l < s->vq_start_subband[k]; l++) {

            int m;



            /* Select the mid-tread linear quantizer */

            int abits = s->bitalloc[k][l];



            float quant_step_size = quant_step_table[abits];



            /*

             * Determine quantization index code book and its type

             */



            /* Select quantization index code book */

            int sel = s->quant_index_huffman[k][abits];



            /*

             * Extract bits from the bit stream

             */

            if (!abits) {

                rscale[l] = 0;

                memset(block + 8 * l, 0, 8 * sizeof(block[0]));

            } else {

                /* Deal with transients */

                int sfi = s->transition_mode[k][l] && subsubframe >= s->transition_mode[k][l];

                rscale[l] = quant_step_size * s->scale_factor[k][l][sfi] *

                               s->scalefactor_adj[k][sel];



                if (abits >= 11 || !dca_smpl_bitalloc[abits].vlc[sel].table) {

                    if (abits <= 7) {

                        /* Block code */

                        int block_code1, block_code2, size, levels, err;



                        size   = abits_sizes[abits - 1];

                        levels = abits_levels[abits - 1];



                        block_code1 = get_bits(&s->gb, size);

                        block_code2 = get_bits(&s->gb, size);

                        err = decode_blockcodes(block_code1, block_code2,

                                                levels, block + 8 * l);

                        if (err) {

                            av_log(s->avctx, AV_LOG_ERROR,

                                   "ERROR: block code look-up failed\n");

                            return AVERROR_INVALIDDATA;

                        }

                    } else {

                        /* no coding */

                        for (m = 0; m < 8; m++)

                            block[8 * l + m] = get_sbits(&s->gb, abits - 3);

                    }

                } else {

                    /* Huffman coded */

                    for (m = 0; m < 8; m++)

                        block[8 * l + m] = get_bitalloc(&s->gb,

                                                &dca_smpl_bitalloc[abits], sel);

                }



            }

        }



        s->fmt_conv.int32_to_float_fmul_array8(&s->fmt_conv, subband_samples[k][0],

                                               block, rscale, 8 * s->vq_start_subband[k]);



        for (l = 0; l < s->vq_start_subband[k]; l++) {

            int m;

            /*

             * Inverse ADPCM if in prediction mode

             */

            if (s->prediction_mode[k][l]) {

                int n;

                if (s->predictor_history)

                    subband_samples[k][l][0] += (adpcm_vb[s->prediction_vq[k][l]][0] *

                                                 s->subband_samples_hist[k][l][3] +

                                                 adpcm_vb[s->prediction_vq[k][l]][1] *

                                                 s->subband_samples_hist[k][l][2] +

                                                 adpcm_vb[s->prediction_vq[k][l]][2] *

                                                 s->subband_samples_hist[k][l][1] +

                                                 adpcm_vb[s->prediction_vq[k][l]][3] *

                                                 s->subband_samples_hist[k][l][0]) *

                                                (1.0f / 8192);

                for (m = 1; m < 8; m++) {

                    float sum = adpcm_vb[s->prediction_vq[k][l]][0] *

                                subband_samples[k][l][m - 1];

                    for (n = 2; n <= 4; n++)

                        if (m >= n)

                            sum += adpcm_vb[s->prediction_vq[k][l]][n - 1] *

                                   subband_samples[k][l][m - n];

                        else if (s->predictor_history)

                            sum += adpcm_vb[s->prediction_vq[k][l]][n - 1] *

                                   s->subband_samples_hist[k][l][m - n + 4];

                    subband_samples[k][l][m] += sum * 1.0f / 8192;

                }

            }

        }



        /*

         * Decode VQ encoded high frequencies

         */

        for (l = s->vq_start_subband[k]; l < s->subband_activity[k]; l++) {

            /* 1 vector -> 32 samples but we only need the 8 samples

             * for this subsubframe. */

            int hfvq = s->high_freq_vq[k][l];



            if (!s->debug_flag & 0x01) {

                av_log(s->avctx, AV_LOG_DEBUG,

                       "Stream with high frequencies VQ coding\n");

                s->debug_flag |= 0x01;

            }



            int8x8_fmul_int32(&s->dcadsp, subband_samples[k][l],

                              &high_freq_vq[hfvq][subsubframe * 8],

                              s->scale_factor[k][l][0]);

        }

    }



    /* Check for DSYNC after subsubframe */

    if (s->aspf || subsubframe == s->subsubframes[s->current_subframe] - 1) {

        if (0xFFFF == get_bits(&s->gb, 16)) {   /* 0xFFFF */

#ifdef TRACE

            av_log(s->avctx, AV_LOG_DEBUG, "Got subframe DSYNC\n");

#endif

        } else {

            av_log(s->avctx, AV_LOG_ERROR, "Didn't get subframe DSYNC\n");

            return AVERROR_INVALIDDATA;

        }

    }



    /* Backup predictor history for adpcm */

    for (k = base_channel; k < s->prim_channels; k++)

        for (l = 0; l < s->vq_start_subband[k]; l++)

            AV_COPY128(s->subband_samples_hist[k][l], &subband_samples[k][l][4]);



    return 0;

}
