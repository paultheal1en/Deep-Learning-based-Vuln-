static int decode_subframe(WMAProDecodeCtx *s)

{

    int offset = s->samples_per_frame;

    int subframe_len = s->samples_per_frame;

    int i;

    int total_samples   = s->samples_per_frame * s->num_channels;

    int transmit_coeffs = 0;

    int cur_subwoofer_cutoff;



    s->subframe_offset = get_bits_count(&s->gb);



    /** reset channel context and find the next block offset and size

        == the next block of the channel with the smallest number of

        decoded samples

    */

    for (i = 0; i < s->num_channels; i++) {

        s->channel[i].grouped = 0;

        if (offset > s->channel[i].decoded_samples) {

            offset = s->channel[i].decoded_samples;

            subframe_len =

                s->channel[i].subframe_len[s->channel[i].cur_subframe];

        }

    }



    av_dlog(s->avctx,

            "processing subframe with offset %i len %i\n", offset, subframe_len);



    /** get a list of all channels that contain the estimated block */

    s->channels_for_cur_subframe = 0;

    for (i = 0; i < s->num_channels; i++) {

        const int cur_subframe = s->channel[i].cur_subframe;

        /** substract already processed samples */

        total_samples -= s->channel[i].decoded_samples;



        /** and count if there are multiple subframes that match our profile */

        if (offset == s->channel[i].decoded_samples &&

            subframe_len == s->channel[i].subframe_len[cur_subframe]) {

            total_samples -= s->channel[i].subframe_len[cur_subframe];

            s->channel[i].decoded_samples +=

                s->channel[i].subframe_len[cur_subframe];

            s->channel_indexes_for_cur_subframe[s->channels_for_cur_subframe] = i;

            ++s->channels_for_cur_subframe;

        }

    }



    /** check if the frame will be complete after processing the

        estimated block */

    if (!total_samples)

        s->parsed_all_subframes = 1;





    av_dlog(s->avctx, "subframe is part of %i channels\n",

            s->channels_for_cur_subframe);



    /** calculate number of scale factor bands and their offsets */

    s->table_idx         = av_log2(s->samples_per_frame/subframe_len);

    s->num_bands         = s->num_sfb[s->table_idx];

    s->cur_sfb_offsets   = s->sfb_offsets[s->table_idx];

    cur_subwoofer_cutoff = s->subwoofer_cutoffs[s->table_idx];



    /** configure the decoder for the current subframe */

    for (i = 0; i < s->channels_for_cur_subframe; i++) {

        int c = s->channel_indexes_for_cur_subframe[i];



        s->channel[c].coeffs = &s->channel[c].out[(s->samples_per_frame >> 1)

                                                  + offset];

    }



    s->subframe_len = subframe_len;

    s->esc_len = av_log2(s->subframe_len - 1) + 1;



    /** skip extended header if any */

    if (get_bits1(&s->gb)) {

        int num_fill_bits;

        if (!(num_fill_bits = get_bits(&s->gb, 2))) {

            int len = get_bits(&s->gb, 4);

            num_fill_bits = get_bits(&s->gb, len) + 1;

        }



        if (num_fill_bits >= 0) {

            if (get_bits_count(&s->gb) + num_fill_bits > s->num_saved_bits) {

                av_log(s->avctx, AV_LOG_ERROR, "invalid number of fill bits\n");

                return AVERROR_INVALIDDATA;

            }



            skip_bits_long(&s->gb, num_fill_bits);

        }

    }



    /** no idea for what the following bit is used */

    if (get_bits1(&s->gb)) {

        av_log_ask_for_sample(s->avctx, "reserved bit set\n");

        return AVERROR_INVALIDDATA;

    }





    if (decode_channel_transform(s) < 0)

        return AVERROR_INVALIDDATA;





    for (i = 0; i < s->channels_for_cur_subframe; i++) {

        int c = s->channel_indexes_for_cur_subframe[i];

        if ((s->channel[c].transmit_coefs = get_bits1(&s->gb)))

            transmit_coeffs = 1;

    }



    if (transmit_coeffs) {

        int step;

        int quant_step = 90 * s->bits_per_sample >> 4;



        /** decode number of vector coded coefficients */

        if ((s->transmit_num_vec_coeffs = get_bits1(&s->gb))) {

            int num_bits = av_log2((s->subframe_len + 3)/4) + 1;

            for (i = 0; i < s->channels_for_cur_subframe; i++) {

                int c = s->channel_indexes_for_cur_subframe[i];

                s->channel[c].num_vec_coeffs = get_bits(&s->gb, num_bits) << 2;

            }

        } else {

            for (i = 0; i < s->channels_for_cur_subframe; i++) {

                int c = s->channel_indexes_for_cur_subframe[i];

                s->channel[c].num_vec_coeffs = s->subframe_len;

            }

        }

        /** decode quantization step */

        step = get_sbits(&s->gb, 6);

        quant_step += step;

        if (step == -32 || step == 31) {

            const int sign = (step == 31) - 1;

            int quant = 0;

            while (get_bits_count(&s->gb) + 5 < s->num_saved_bits &&

                   (step = get_bits(&s->gb, 5)) == 31) {

                quant += 31;

            }

            quant_step += ((quant + step) ^ sign) - sign;

        }

        if (quant_step < 0) {

            av_log(s->avctx, AV_LOG_DEBUG, "negative quant step\n");

        }



        /** decode quantization step modifiers for every channel */



        if (s->channels_for_cur_subframe == 1) {

            s->channel[s->channel_indexes_for_cur_subframe[0]].quant_step = quant_step;

        } else {

            int modifier_len = get_bits(&s->gb, 3);

            for (i = 0; i < s->channels_for_cur_subframe; i++) {

                int c = s->channel_indexes_for_cur_subframe[i];

                s->channel[c].quant_step = quant_step;

                if (get_bits1(&s->gb)) {

                    if (modifier_len) {

                        s->channel[c].quant_step += get_bits(&s->gb, modifier_len) + 1;

                    } else

                        ++s->channel[c].quant_step;

                }

            }

        }



        /** decode scale factors */

        if (decode_scale_factors(s) < 0)

            return AVERROR_INVALIDDATA;

    }



    av_dlog(s->avctx, "BITSTREAM: subframe header length was %i\n",

            get_bits_count(&s->gb) - s->subframe_offset);



    /** parse coefficients */

    for (i = 0; i < s->channels_for_cur_subframe; i++) {

        int c = s->channel_indexes_for_cur_subframe[i];

        if (s->channel[c].transmit_coefs &&

            get_bits_count(&s->gb) < s->num_saved_bits) {

            decode_coeffs(s, c);

        } else

            memset(s->channel[c].coeffs, 0,

                   sizeof(*s->channel[c].coeffs) * subframe_len);

    }



    av_dlog(s->avctx, "BITSTREAM: subframe length was %i\n",

            get_bits_count(&s->gb) - s->subframe_offset);



    if (transmit_coeffs) {

        FFTContext *mdct = &s->mdct_ctx[av_log2(subframe_len) - WMAPRO_BLOCK_MIN_BITS];

        /** reconstruct the per channel data */

        inverse_channel_transform(s);

        for (i = 0; i < s->channels_for_cur_subframe; i++) {

            int c = s->channel_indexes_for_cur_subframe[i];

            const int* sf = s->channel[c].scale_factors;

            int b;



            if (c == s->lfe_channel)

                memset(&s->tmp[cur_subwoofer_cutoff], 0, sizeof(*s->tmp) *

                       (subframe_len - cur_subwoofer_cutoff));



            /** inverse quantization and rescaling */

            for (b = 0; b < s->num_bands; b++) {

                const int end = FFMIN(s->cur_sfb_offsets[b+1], s->subframe_len);

                const int exp = s->channel[c].quant_step -

                            (s->channel[c].max_scale_factor - *sf++) *

                            s->channel[c].scale_factor_step;

                const float quant = pow(10.0, exp / 20.0);

                int start = s->cur_sfb_offsets[b];

                s->dsp.vector_fmul_scalar(s->tmp + start,

                                          s->channel[c].coeffs + start,

                                          quant, end - start);

            }



            /** apply imdct (imdct_half == DCTIV with reverse) */

            mdct->imdct_half(mdct, s->channel[c].coeffs, s->tmp);

        }

    }



    /** window and overlapp-add */

    wmapro_window(s);



    /** handled one subframe */

    for (i = 0; i < s->channels_for_cur_subframe; i++) {

        int c = s->channel_indexes_for_cur_subframe[i];

        if (s->channel[c].cur_subframe >= s->channel[c].num_subframes) {

            av_log(s->avctx, AV_LOG_ERROR, "broken subframe\n");

            return AVERROR_INVALIDDATA;

        }

        ++s->channel[c].cur_subframe;

    }



    return 0;

}
