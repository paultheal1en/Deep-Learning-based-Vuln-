static int decode_subframe(WmallDecodeCtx *s)

{

    int offset        = s->samples_per_frame;

    int subframe_len  = s->samples_per_frame;

    int total_samples = s->samples_per_frame * s->num_channels;

    int i, j, rawpcm_tile, padding_zeroes, res;



    s->subframe_offset = get_bits_count(&s->gb);



    /* reset channel context and find the next block offset and size

        == the next block of the channel with the smallest number of

        decoded samples */

    for (i = 0; i < s->num_channels; i++) {

        if (offset > s->channel[i].decoded_samples) {

            offset = s->channel[i].decoded_samples;

            subframe_len =

                s->channel[i].subframe_len[s->channel[i].cur_subframe];

        }

    }



    /* get a list of all channels that contain the estimated block */

    s->channels_for_cur_subframe = 0;

    for (i = 0; i < s->num_channels; i++) {

        const int cur_subframe = s->channel[i].cur_subframe;

        /* subtract already processed samples */

        total_samples -= s->channel[i].decoded_samples;



        /* and count if there are multiple subframes that match our profile */

        if (offset == s->channel[i].decoded_samples &&

            subframe_len == s->channel[i].subframe_len[cur_subframe]) {

            total_samples -= s->channel[i].subframe_len[cur_subframe];

            s->channel[i].decoded_samples +=

                s->channel[i].subframe_len[cur_subframe];

            s->channel_indexes_for_cur_subframe[s->channels_for_cur_subframe] = i;

            ++s->channels_for_cur_subframe;

        }

    }



    /* check if the frame will be complete after processing the

        estimated block */

    if (!total_samples)

        s->parsed_all_subframes = 1;





    s->seekable_tile = get_bits1(&s->gb);

    if (s->seekable_tile) {

        clear_codec_buffers(s);



        s->do_arith_coding    = get_bits1(&s->gb);

        if (s->do_arith_coding) {

            avpriv_request_sample(s->avctx, "Arithmetic coding");

            return AVERROR_PATCHWELCOME;

        }

        s->do_ac_filter       = get_bits1(&s->gb);

        s->do_inter_ch_decorr = get_bits1(&s->gb);

        s->do_mclms           = get_bits1(&s->gb);



        if (s->do_ac_filter)

            decode_ac_filter(s);



        if (s->do_mclms)

            decode_mclms(s);



        if ((res = decode_cdlms(s)) < 0)

            return res;

        s->movave_scaling = get_bits(&s->gb, 3);

        s->quant_stepsize = get_bits(&s->gb, 8) + 1;



        reset_codec(s);

    } else if (!s->cdlms[0][0].order) {

        av_log(s->avctx, AV_LOG_DEBUG,

               "Waiting for seekable tile\n");

        s->frame.nb_samples = 0;

        return -1;

    }



    rawpcm_tile = get_bits1(&s->gb);



    for (i = 0; i < s->num_channels; i++)

        s->is_channel_coded[i] = 1;



    if (!rawpcm_tile) {

        for (i = 0; i < s->num_channels; i++)

            s->is_channel_coded[i] = get_bits1(&s->gb);



        if (s->bV3RTM) {

            // LPC

            s->do_lpc = get_bits1(&s->gb);

            if (s->do_lpc) {

                decode_lpc(s);

                avpriv_request_sample(s->avctx, "Expect wrong output since "

                                      "inverse LPC filter");

            }

        } else

            s->do_lpc = 0;

    }





    if (get_bits1(&s->gb))

        padding_zeroes = get_bits(&s->gb, 5);

    else

        padding_zeroes = 0;



    if (rawpcm_tile) {

        int bits = s->bits_per_sample - padding_zeroes;

        if (bits <= 0) {

            av_log(s->avctx, AV_LOG_ERROR,

                   "Invalid number of padding bits in raw PCM tile\n");

            return AVERROR_INVALIDDATA;

        }

        av_dlog(s->avctx, "RAWPCM %d bits per sample. "

                "total %d bits, remain=%d\n", bits,

                bits * s->num_channels * subframe_len, get_bits_count(&s->gb));

        for (i = 0; i < s->num_channels; i++)

            for (j = 0; j < subframe_len; j++)

                s->channel_coeffs[i][j] = get_sbits(&s->gb, bits);

    } else {

        for (i = 0; i < s->num_channels; i++)

            if (s->is_channel_coded[i]) {

                decode_channel_residues(s, i, subframe_len);

                if (s->seekable_tile)

                    use_high_update_speed(s, i);

                else

                    use_normal_update_speed(s, i);

                revert_cdlms(s, i, 0, subframe_len);

            } else {

                memset(s->channel_residues[i], 0, sizeof(**s->channel_residues) * subframe_len);

            }

    }

    if (s->do_mclms)

        revert_mclms(s, subframe_len);

    if (s->do_inter_ch_decorr)

        revert_inter_ch_decorr(s, subframe_len);

    if (s->do_ac_filter)

        revert_acfilter(s, subframe_len);



    /* Dequantize */

    if (s->quant_stepsize != 1)

        for (i = 0; i < s->num_channels; i++)

            for (j = 0; j < subframe_len; j++)

                s->channel_residues[i][j] *= s->quant_stepsize;



    /* Write to proper output buffer depending on bit-depth */

    for (i = 0; i < s->channels_for_cur_subframe; i++) {

        int c = s->channel_indexes_for_cur_subframe[i];

        int subframe_len = s->channel[c].subframe_len[s->channel[c].cur_subframe];



        for (j = 0; j < subframe_len; j++) {

            if (s->bits_per_sample == 16) {

                *s->samples_16[c]++ = (int16_t) s->channel_residues[c][j] << padding_zeroes;

            } else {

                *s->samples_32[c]++ = s->channel_residues[c][j] << padding_zeroes;

            }

        }

    }



    /* handled one subframe */

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
