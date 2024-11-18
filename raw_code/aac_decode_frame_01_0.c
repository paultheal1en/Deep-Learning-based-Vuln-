static int aac_decode_frame(AVCodecContext * avccontext, void * data, int * data_size, const uint8_t * buf, int buf_size) {

    AACContext * ac = avccontext->priv_data;

    GetBitContext gb;

    enum RawDataBlockType elem_type;

    int err, elem_id, data_size_tmp;



    init_get_bits(&gb, buf, buf_size*8);



    if (show_bits(&gb, 12) == 0xfff) {

        if ((err = parse_adts_frame_header(ac, &gb)) < 0) {

            av_log(avccontext, AV_LOG_ERROR, "Error decoding AAC frame header.\n");

            return -1;

        }

        if (ac->m4ac.sampling_index > 11) {

            av_log(ac->avccontext, AV_LOG_ERROR, "invalid sampling rate index %d\n", ac->m4ac.sampling_index);

            return -1;

        }

    }



    // parse

    while ((elem_type = get_bits(&gb, 3)) != TYPE_END) {

        elem_id = get_bits(&gb, 4);

        err = -1;



        if(elem_type == TYPE_SCE && elem_id == 1 &&

                !ac->che[TYPE_SCE][elem_id] && ac->che[TYPE_LFE][0]) {

            /* Some streams incorrectly code 5.1 audio as SCE[0] CPE[0] CPE[1] SCE[1]

               instead of SCE[0] CPE[0] CPE[0] LFE[0]. If we seem to have

               encountered such a stream, transfer the LFE[0] element to SCE[1] */

            ac->che[TYPE_SCE][elem_id] = ac->che[TYPE_LFE][0];

            ac->che[TYPE_LFE][0] = NULL;

        }

        if(elem_type < TYPE_DSE) {

            if(!ac->che[elem_type][elem_id])

                return -1;

            if(elem_type != TYPE_CCE)

                ac->che[elem_type][elem_id]->coup.coupling_point = 4;

        }



        switch (elem_type) {



        case TYPE_SCE:

            err = decode_ics(ac, &ac->che[TYPE_SCE][elem_id]->ch[0], &gb, 0, 0);

            break;



        case TYPE_CPE:

            err = decode_cpe(ac, &gb, elem_id);

            break;



        case TYPE_CCE:

            err = decode_cce(ac, &gb, ac->che[TYPE_CCE][elem_id]);

            break;



        case TYPE_LFE:

            err = decode_ics(ac, &ac->che[TYPE_LFE][elem_id]->ch[0], &gb, 0, 0);

            break;



        case TYPE_DSE:

            skip_data_stream_element(&gb);

            err = 0;

            break;



        case TYPE_PCE:

        {

            enum ChannelPosition new_che_pos[4][MAX_ELEM_ID];

            memset(new_che_pos, 0, 4 * MAX_ELEM_ID * sizeof(new_che_pos[0][0]));

            if((err = decode_pce(ac, new_che_pos, &gb)))

                break;

            err = output_configure(ac, ac->che_pos, new_che_pos);

            break;

        }



        case TYPE_FIL:

            if (elem_id == 15)

                elem_id += get_bits(&gb, 8) - 1;

            while (elem_id > 0)

                elem_id -= decode_extension_payload(ac, &gb, elem_id);

            err = 0; /* FIXME */

            break;



        default:

            err = -1; /* should not happen, but keeps compiler happy */

            break;

        }



        if(err)

            return err;

    }



    spectral_to_sample(ac);



    if (!ac->is_saved) {

        ac->is_saved = 1;

        *data_size = 0;

        return buf_size;

    }



    data_size_tmp = 1024 * avccontext->channels * sizeof(int16_t);

    if(*data_size < data_size_tmp) {

        av_log(avccontext, AV_LOG_ERROR,

               "Output buffer too small (%d) or trying to output too many samples (%d) for this frame.\n",

               *data_size, data_size_tmp);

        return -1;

    }

    *data_size = data_size_tmp;



    ac->dsp.float_to_int16_interleave(data, (const float **)ac->output_data, 1024, avccontext->channels);



    return buf_size;

}
