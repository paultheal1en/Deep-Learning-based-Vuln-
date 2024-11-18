static int aac_decode_frame_int(AVCodecContext *avctx, void *data,

                                int *got_frame_ptr, GetBitContext *gb)

{

    AACContext *ac = avctx->priv_data;

    ChannelElement *che = NULL, *che_prev = NULL;

    enum RawDataBlockType elem_type, elem_type_prev = TYPE_END;

    int err, elem_id;

    int samples = 0, multiplier, audio_found = 0, pce_found = 0;



    if (show_bits(gb, 12) == 0xfff) {

        if (parse_adts_frame_header(ac, gb) < 0) {

            av_log(avctx, AV_LOG_ERROR, "Error decoding AAC frame header.\n");

            err = -1;

            goto fail;

        }

        if (ac->oc[1].m4ac.sampling_index > 12) {

            av_log(ac->avctx, AV_LOG_ERROR, "invalid sampling rate index %d\n", ac->oc[1].m4ac.sampling_index);

            err = -1;

            goto fail;

        }

    }



    ac->tags_mapped = 0;

    // parse

    while ((elem_type = get_bits(gb, 3)) != TYPE_END) {

        elem_id = get_bits(gb, 4);



        if (elem_type < TYPE_DSE) {

            if (!(che=get_che(ac, elem_type, elem_id))) {

                av_log(ac->avctx, AV_LOG_ERROR, "channel element %d.%d is not allocated\n",

                       elem_type, elem_id);

                err = -1;

                goto fail;

            }

            samples = 1024;

        }



        switch (elem_type) {



        case TYPE_SCE:

            err = decode_ics(ac, &che->ch[0], gb, 0, 0);

            audio_found = 1;

            break;



        case TYPE_CPE:

            err = decode_cpe(ac, gb, che);

            audio_found = 1;

            break;



        case TYPE_CCE:

            err = decode_cce(ac, gb, che);

            break;



        case TYPE_LFE:

            err = decode_ics(ac, &che->ch[0], gb, 0, 0);

            audio_found = 1;

            break;



        case TYPE_DSE:

            err = skip_data_stream_element(ac, gb);

            break;



        case TYPE_PCE: {

            uint8_t layout_map[MAX_ELEM_ID*4][3];

            int tags;

            push_output_configuration(ac);

            tags = decode_pce(avctx, &ac->oc[1].m4ac, layout_map, gb);

            if (tags < 0) {

                err = tags;

                break;

            }

            if (pce_found) {

                av_log(avctx, AV_LOG_ERROR,

                       "Not evaluating a further program_config_element as this construct is dubious at best.\n");

                pop_output_configuration(ac);

            } else {

                err = output_configure(ac, layout_map, tags, 0, OC_TRIAL_PCE);

                if (!err)

                    ac->oc[1].m4ac.chan_config = 0;

                pce_found = 1;

            }

            break;

        }



        case TYPE_FIL:

            if (elem_id == 15)

                elem_id += get_bits(gb, 8) - 1;

            if (get_bits_left(gb) < 8 * elem_id) {

                    av_log(avctx, AV_LOG_ERROR, overread_err);

                    err = -1;

                    goto fail;

            }

            while (elem_id > 0)

                elem_id -= decode_extension_payload(ac, gb, elem_id, che_prev, elem_type_prev);

            err = 0; /* FIXME */

            break;



        default:

            err = -1; /* should not happen, but keeps compiler happy */

            break;

        }



        che_prev       = che;

        elem_type_prev = elem_type;



        if (err)

            goto fail;



        if (get_bits_left(gb) < 3) {

            av_log(avctx, AV_LOG_ERROR, overread_err);

            err = -1;

            goto fail;

        }

    }



    spectral_to_sample(ac);



    multiplier = (ac->oc[1].m4ac.sbr == 1) ? ac->oc[1].m4ac.ext_sample_rate > ac->oc[1].m4ac.sample_rate : 0;

    samples <<= multiplier;



    if (samples) {

        /* get output buffer */

        ac->frame.nb_samples = samples;

        if ((err = avctx->get_buffer(avctx, &ac->frame)) < 0) {

            av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

            err = -1;

            goto fail;

        }



        if (avctx->sample_fmt == AV_SAMPLE_FMT_FLT)

            ac->fmt_conv.float_interleave((float *)ac->frame.data[0],

                                          (const float **)ac->output_data,

                                          samples, avctx->channels);

        else

            ac->fmt_conv.float_to_int16_interleave((int16_t *)ac->frame.data[0],

                                                   (const float **)ac->output_data,

                                                   samples, avctx->channels);



        *(AVFrame *)data = ac->frame;

    }

    *got_frame_ptr = !!samples;



    if (ac->oc[1].status && audio_found) {

        avctx->sample_rate = ac->oc[1].m4ac.sample_rate << multiplier;

        avctx->frame_size = samples;

        ac->oc[1].status = OC_LOCKED;

    }



    return 0;

fail:

    pop_output_configuration(ac);

    return err;

}
