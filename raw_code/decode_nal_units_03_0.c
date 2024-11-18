static int decode_nal_units(HEVCContext *s, const uint8_t *buf, int length)

{

    int i, consumed, ret = 0;



    s->ref = NULL;

    s->eos = 0;



    /* split the input packet into NAL units, so we know the upper bound on the

     * number of slices in the frame */

    s->nb_nals = 0;

    while (length >= 4) {

        HEVCNAL *nal;

        int extract_length = 0;



        if (s->is_nalff) {

            int i;

            for (i = 0; i < s->nal_length_size; i++)

                extract_length = (extract_length << 8) | buf[i];

            buf    += s->nal_length_size;

            length -= s->nal_length_size;



            if (extract_length > length) {

                av_log(s->avctx, AV_LOG_ERROR, "Invalid NAL unit size.\n");

                ret = AVERROR_INVALIDDATA;

                goto fail;

            }

        } else {

            if (buf[2] == 0) {

                length--;

                buf++;

                continue;

            }

            if (buf[0] != 0 || buf[1] != 0 || buf[2] != 1) {

                ret = AVERROR_INVALIDDATA;

                goto fail;

            }



            buf    += 3;

            length -= 3;

        }



        if (!s->is_nalff)

            extract_length = length;



        if (s->nals_allocated < s->nb_nals + 1) {

            int new_size = s->nals_allocated + 1;

            HEVCNAL *tmp = av_realloc_array(s->nals, new_size, sizeof(*tmp));

            if (!tmp) {

                ret = AVERROR(ENOMEM);

                goto fail;

            }

            s->nals = tmp;

            memset(s->nals + s->nals_allocated, 0, (new_size - s->nals_allocated) * sizeof(*tmp));

            av_reallocp_array(&s->skipped_bytes_nal, new_size, sizeof(*s->skipped_bytes_nal));

            av_reallocp_array(&s->skipped_bytes_pos_size_nal, new_size, sizeof(*s->skipped_bytes_pos_size_nal));

            av_reallocp_array(&s->skipped_bytes_pos_nal, new_size, sizeof(*s->skipped_bytes_pos_nal));

            s->skipped_bytes_pos_size_nal[s->nals_allocated] = 1024; // initial buffer size

            s->skipped_bytes_pos_nal[s->nals_allocated] = av_malloc_array(s->skipped_bytes_pos_size_nal[s->nals_allocated], sizeof(*s->skipped_bytes_pos));

            s->nals_allocated = new_size;

        }

        s->skipped_bytes_pos_size = s->skipped_bytes_pos_size_nal[s->nb_nals];

        s->skipped_bytes_pos = s->skipped_bytes_pos_nal[s->nb_nals];

        nal = &s->nals[s->nb_nals];



        consumed = extract_rbsp(s, buf, extract_length, nal);



        s->skipped_bytes_nal[s->nb_nals] = s->skipped_bytes;

        s->skipped_bytes_pos_size_nal[s->nb_nals] = s->skipped_bytes_pos_size;

        s->skipped_bytes_pos_nal[s->nb_nals++] = s->skipped_bytes_pos;





        if (consumed < 0) {

            ret = consumed;

            goto fail;

        }



        ret = init_get_bits8(&s->HEVClc->gb, nal->data, nal->size);

        if (ret < 0)

            goto fail;

        hls_nal_unit(s);



        if (s->nal_unit_type == NAL_EOS_NUT || s->nal_unit_type == NAL_EOB_NUT)

            s->eos = 1;



        buf    += consumed;

        length -= consumed;

    }



    /* parse the NAL units */

    for (i = 0; i < s->nb_nals; i++) {

        int ret;

        s->skipped_bytes = s->skipped_bytes_nal[i];

        s->skipped_bytes_pos = s->skipped_bytes_pos_nal[i];



        ret = decode_nal_unit(s, s->nals[i].data, s->nals[i].size);

        if (ret < 0) {

            av_log(s->avctx, AV_LOG_WARNING, "Error parsing NAL unit #%d.\n", i);

            if (s->avctx->err_recognition & AV_EF_EXPLODE)

                goto fail;

        }

    }



fail:

    if (s->ref && s->threads_type == FF_THREAD_FRAME)

        ff_thread_report_progress(&s->ref->tf, INT_MAX, 0);



    return ret;

}
