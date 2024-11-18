static int aac_decode_frame_int(AVCodecContext *avctx, void *data,

                                int *got_frame_ptr, GetBitContext *gb, AVPacket *avpkt)

{

    AACContext *ac = avctx->priv_data;

    ChannelElement *che = NULL, *che_prev = NULL;

    enum RawDataBlockType elem_type, che_prev_type = TYPE_END;

    int err, elem_id;

    int samples = 0, multiplier, audio_found = 0, pce_found = 0;

    int is_dmono, sce_count = 0;

    int payload_alignment;



    ac->frame = data;



    if (show_bits(gb, 12) == 0xfff) {

        if ((err = parse_adts_frame_header(ac, gb)) < 0) {

            av_log(avctx, AV_LOG_ERROR, "Error decoding AAC frame header.\n");

            goto fail;

        }

        if (ac->oc[1].m4ac.sampling_index > 12) {

            av_log(ac->avctx, AV_LOG_ERROR, "invalid sampling rate index %d\n", ac->oc[1].m4ac.sampling_index);

            err = AVERROR_INVALIDDATA;

            goto fail;

        }

    }



    if ((err = frame_configure_elements(avctx)) < 0)

        goto fail;



    // The FF_PROFILE_AAC_* defines are all object_type - 1

    // This may lead to an undefined profile being signaled

    ac->avctx->profile = ac->oc[1].m4ac.object_type - 1;



    payload_alignment = get_bits_count(gb);

    ac->tags_mapped = 0;

    // parse

    while ((elem_type = get_bits(gb, 3)) != TYPE_END) {

        elem_id = get_bits(gb, 4);



        if (avctx->debug & FF_DEBUG_STARTCODE)

            av_log(avctx, AV_LOG_DEBUG, "Elem type:%x id:%x\n", elem_type, elem_id);



        if (!avctx->channels && elem_type != TYPE_PCE) {

            err = AVERROR_INVALIDDATA;

            goto fail;

        }



        if (elem_type < TYPE_DSE) {

            if (!(che=get_che(ac, elem_type, elem_id))) {

                av_log(ac->avctx, AV_LOG_ERROR, "channel element %d.%d is not allocated\n",

                       elem_type, elem_id);

                err = AVERROR_INVALIDDATA;

                goto fail;

            }

            samples = 1024;

            che->present = 1;

        }



        switch (elem_type) {



        case TYPE_SCE:

            err = decode_ics(ac, &che->ch[0], gb, 0, 0);

            audio_found = 1;

            sce_count++;

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

            tags = decode_pce(avctx, &ac->oc[1].m4ac, layout_map, gb,

                              payload_alignment);

            if (tags < 0) {

                err = tags;

                break;

            }

            if (pce_found) {

                av_log(avctx, AV_LOG_ERROR,

                       "Not evaluating a further program_config_element as this construct is dubious at best.\n");

                pop_output_configuration(ac);

            } else {

                err = output_configure(ac, layout_map, tags, OC_TRIAL_PCE, 1);

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

                    av_log(avctx, AV_LOG_ERROR, "TYPE_FIL: "overread_err);

                    err = AVERROR_INVALIDDATA;

                    goto fail;

            }

            while (elem_id > 0)

                elem_id -= decode_extension_payload(ac, gb, elem_id, che_prev, che_prev_type);

            err = 0; /* FIXME */

            break;



        default:

            err = AVERROR_BUG; /* should not happen, but keeps compiler happy */

            break;

        }



        if (elem_type < TYPE_DSE) {

            che_prev      = che;

            che_prev_type = elem_type;

        }



        if (err)

            goto fail;



        if (get_bits_left(gb) < 3) {

            av_log(avctx, AV_LOG_ERROR, overread_err);

            err = AVERROR_INVALIDDATA;

            goto fail;

        }

    }



    if (!avctx->channels) {

        *got_frame_ptr = 0;

        return 0;

    }



    multiplier = (ac->oc[1].m4ac.sbr == 1) ? ac->oc[1].m4ac.ext_sample_rate > ac->oc[1].m4ac.sample_rate : 0;

    samples <<= multiplier;



    spectral_to_sample(ac, samples);



    if (ac->oc[1].status && audio_found) {

        avctx->sample_rate = ac->oc[1].m4ac.sample_rate << multiplier;

        avctx->frame_size = samples;

        ac->oc[1].status = OC_LOCKED;

    }



    if (multiplier)

        avctx->internal->skip_samples_multiplier = 2;



    if (!ac->frame->data[0] && samples) {

        av_log(avctx, AV_LOG_ERROR, "no frame data found\n");

        err = AVERROR_INVALIDDATA;

        goto fail;

    }



    if (samples) {

        ac->frame->nb_samples = samples;

        ac->frame->sample_rate = avctx->sample_rate;

    } else

        av_frame_unref(ac->frame);

    *got_frame_ptr = !!samples;



    /* for dual-mono audio (SCE + SCE) */

    is_dmono = ac->dmono_mode && sce_count == 2 &&

               ac->oc[1].channel_layout == (AV_CH_FRONT_LEFT | AV_CH_FRONT_RIGHT);

    if (is_dmono) {

        if (ac->dmono_mode == 1)

            ((AVFrame *)data)->data[1] =((AVFrame *)data)->data[0];

        else if (ac->dmono_mode == 2)

            ((AVFrame *)data)->data[0] =((AVFrame *)data)->data[1];

    }



    return 0;

fail:

    pop_output_configuration(ac);

    return err;

}
