static int aac_decode_frame(AVCodecContext *avccontext, void *data,

                            int *data_size, AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    AACContext *ac = avccontext->priv_data;

    ChannelElement *che = NULL, *che_prev = NULL;

    GetBitContext gb;

    enum RawDataBlockType elem_type, elem_type_prev = TYPE_END;

    int err, elem_id, data_size_tmp;

    int buf_consumed;

    int samples = 1024, multiplier;

    int buf_offset;



    init_get_bits(&gb, buf, buf_size * 8);



    if (show_bits(&gb, 12) == 0xfff) {

        if (parse_adts_frame_header(ac, &gb) < 0) {

            av_log(avccontext, AV_LOG_ERROR, "Error decoding AAC frame header.\n");

            return -1;

        }

        if (ac->m4ac.sampling_index > 12) {

            av_log(ac->avccontext, AV_LOG_ERROR, "invalid sampling rate index %d\n", ac->m4ac.sampling_index);

            return -1;

        }

    }




    // parse

    while ((elem_type = get_bits(&gb, 3)) != TYPE_END) {

        elem_id = get_bits(&gb, 4);



        if (elem_type < TYPE_DSE && !(che=get_che(ac, elem_type, elem_id))) {

            av_log(ac->avccontext, AV_LOG_ERROR, "channel element %d.%d is not allocated\n", elem_type, elem_id);

            return -1;

        }



        switch (elem_type) {



        case TYPE_SCE:

            err = decode_ics(ac, &che->ch[0], &gb, 0, 0);

            break;



        case TYPE_CPE:

            err = decode_cpe(ac, &gb, che);

            break;



        case TYPE_CCE:

            err = decode_cce(ac, &gb, che);

            break;



        case TYPE_LFE:

            err = decode_ics(ac, &che->ch[0], &gb, 0, 0);

            break;



        case TYPE_DSE:

            err = skip_data_stream_element(ac, &gb);

            break;



        case TYPE_PCE: {

            enum ChannelPosition new_che_pos[4][MAX_ELEM_ID];

            memset(new_che_pos, 0, 4 * MAX_ELEM_ID * sizeof(new_che_pos[0][0]));

            if ((err = decode_pce(ac, new_che_pos, &gb)))

                break;

            if (ac->output_configured > OC_TRIAL_PCE)

                av_log(avccontext, AV_LOG_ERROR,

                       "Not evaluating a further program_config_element as this construct is dubious at best.\n");

            else

                err = output_configure(ac, ac->che_pos, new_che_pos, 0, OC_TRIAL_PCE);

            break;

        }



        case TYPE_FIL:

            if (elem_id == 15)

                elem_id += get_bits(&gb, 8) - 1;

            if (get_bits_left(&gb) < 8 * elem_id) {

                    av_log(avccontext, AV_LOG_ERROR, overread_err);

                    return -1;

            }

            while (elem_id > 0)

                elem_id -= decode_extension_payload(ac, &gb, elem_id, che_prev, elem_type_prev);

            err = 0; /* FIXME */

            break;



        default:

            err = -1; /* should not happen, but keeps compiler happy */

            break;

        }



        che_prev       = che;

        elem_type_prev = elem_type;



        if (err)

            return err;



        if (get_bits_left(&gb) < 3) {

            av_log(avccontext, AV_LOG_ERROR, overread_err);

            return -1;

        }

    }



    spectral_to_sample(ac);



    multiplier = (ac->m4ac.sbr == 1) ? ac->m4ac.ext_sample_rate > ac->m4ac.sample_rate : 0;

    samples <<= multiplier;

    if (ac->output_configured < OC_LOCKED) {

        avccontext->sample_rate = ac->m4ac.sample_rate << multiplier;

        avccontext->frame_size = samples;

    }



    data_size_tmp = samples * avccontext->channels * sizeof(int16_t);

    if (*data_size < data_size_tmp) {

        av_log(avccontext, AV_LOG_ERROR,

               "Output buffer too small (%d) or trying to output too many samples (%d) for this frame.\n",

               *data_size, data_size_tmp);

        return -1;

    }

    *data_size = data_size_tmp;



    ac->dsp.float_to_int16_interleave(data, (const float **)ac->output_data, samples, avccontext->channels);



    if (ac->output_configured)

        ac->output_configured = OC_LOCKED;



    buf_consumed = (get_bits_count(&gb) + 7) >> 3;

    for (buf_offset = buf_consumed; buf_offset < buf_size; buf_offset++)

        if (buf[buf_offset])

            break;



    return buf_size > buf_offset ? buf_consumed : buf_size;

}