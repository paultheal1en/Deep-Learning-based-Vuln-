int ff_parse_mpeg2_descriptor(AVFormatContext *fc, AVStream *st, int stream_type,

                              const uint8_t **pp, const uint8_t *desc_list_end,

                              Mp4Descr *mp4_descr, int mp4_descr_count, int pid,

                              MpegTSContext *ts)

{

    const uint8_t *desc_end;

    int desc_len, desc_tag, desc_es_id;

    char language[252];

    int i;



    desc_tag = get8(pp, desc_list_end);

    if (desc_tag < 0)

        return AVERROR_INVALIDDATA;

    desc_len = get8(pp, desc_list_end);

    if (desc_len < 0)

        return AVERROR_INVALIDDATA;

    desc_end = *pp + desc_len;

    if (desc_end > desc_list_end)

        return AVERROR_INVALIDDATA;



    av_dlog(fc, "tag: 0x%02x len=%d\n", desc_tag, desc_len);



    if (st->codec->codec_id == AV_CODEC_ID_NONE &&

        stream_type == STREAM_TYPE_PRIVATE_DATA)

        mpegts_find_stream_type(st, desc_tag, DESC_types);



    switch (desc_tag) {

    case 0x1E: /* SL descriptor */

        desc_es_id = get16(pp, desc_end);

        if (ts && ts->pids[pid])

            ts->pids[pid]->es_id = desc_es_id;

        for (i = 0; i < mp4_descr_count; i++)

            if (mp4_descr[i].dec_config_descr_len &&

                mp4_descr[i].es_id == desc_es_id) {

                AVIOContext pb;

                ffio_init_context(&pb, mp4_descr[i].dec_config_descr,

                                  mp4_descr[i].dec_config_descr_len, 0,

                                  NULL, NULL, NULL, NULL);

                ff_mp4_read_dec_config_descr(fc, st, &pb);

                if (st->codec->codec_id == AV_CODEC_ID_AAC &&

                    st->codec->extradata_size > 0)

                    st->need_parsing = 0;

                if (st->codec->codec_id == AV_CODEC_ID_MPEG4SYSTEMS)

                    mpegts_open_section_filter(ts, pid, m4sl_cb, ts, 1);

            }

        break;

    case 0x1F: /* FMC descriptor */

        get16(pp, desc_end);

        if (mp4_descr_count > 0 &&

            st->codec->codec_id == AV_CODEC_ID_AAC_LATM &&

            mp4_descr->dec_config_descr_len && mp4_descr->es_id == pid) {

            AVIOContext pb;

            ffio_init_context(&pb, mp4_descr->dec_config_descr,

                              mp4_descr->dec_config_descr_len, 0,

                              NULL, NULL, NULL, NULL);

            ff_mp4_read_dec_config_descr(fc, st, &pb);

            if (st->codec->codec_id == AV_CODEC_ID_AAC &&

                st->codec->extradata_size > 0)

                st->need_parsing = 0;

        }

        break;

    case 0x56: /* DVB teletext descriptor */

        language[0] = get8(pp, desc_end);

        language[1] = get8(pp, desc_end);

        language[2] = get8(pp, desc_end);

        language[3] = 0;

        av_dict_set(&st->metadata, "language", language, 0);

        break;

    case 0x59: /* subtitling descriptor */

        language[0] = get8(pp, desc_end);

        language[1] = get8(pp, desc_end);

        language[2] = get8(pp, desc_end);

        language[3] = 0;

        /* hearing impaired subtitles detection */

        switch (get8(pp, desc_end)) {

        case 0x20: /* DVB subtitles (for the hard of hearing) with no monitor aspect ratio criticality */

        case 0x21: /* DVB subtitles (for the hard of hearing) for display on 4:3 aspect ratio monitor */

        case 0x22: /* DVB subtitles (for the hard of hearing) for display on 16:9 aspect ratio monitor */

        case 0x23: /* DVB subtitles (for the hard of hearing) for display on 2.21:1 aspect ratio monitor */

        case 0x24: /* DVB subtitles (for the hard of hearing) for display on a high definition monitor */

        case 0x25: /* DVB subtitles (for the hard of hearing) with plano-stereoscopic disparity for display on a high definition monitor */

            st->disposition |= AV_DISPOSITION_HEARING_IMPAIRED;

            break;

        }

        if (st->codec->extradata) {

            if (st->codec->extradata_size == 4 &&

                memcmp(st->codec->extradata, *pp, 4))

                avpriv_request_sample(fc, "DVB sub with multiple IDs");

        } else {

            st->codec->extradata = av_malloc(4 + FF_INPUT_BUFFER_PADDING_SIZE);

            if (st->codec->extradata) {

                st->codec->extradata_size = 4;

                memcpy(st->codec->extradata, *pp, 4);

            }

        }

        *pp += 4;

        av_dict_set(&st->metadata, "language", language, 0);

        break;

    case 0x0a: /* ISO 639 language descriptor */

        for (i = 0; i + 4 <= desc_len; i += 4) {

            language[i + 0] = get8(pp, desc_end);

            language[i + 1] = get8(pp, desc_end);

            language[i + 2] = get8(pp, desc_end);

            language[i + 3] = ',';

            switch (get8(pp, desc_end)) {

            case 0x01:

                st->disposition |= AV_DISPOSITION_CLEAN_EFFECTS;

                break;

            case 0x02:

                st->disposition |= AV_DISPOSITION_HEARING_IMPAIRED;

                break;

            case 0x03:

                st->disposition |= AV_DISPOSITION_VISUAL_IMPAIRED;

                break;

            }

        }

        if (i && language[0]) {

            language[i - 1] = 0;

            av_dict_set(&st->metadata, "language", language, 0);

        }

        break;

    case 0x05: /* registration descriptor */

        st->codec->codec_tag = bytestream_get_le32(pp);

        av_dlog(fc, "reg_desc=%.4s\n", (char *)&st->codec->codec_tag);

        if (st->codec->codec_id == AV_CODEC_ID_NONE)

            mpegts_find_stream_type(st, st->codec->codec_tag, REGD_types);

        break;

    default:

        break;

    }

    *pp = desc_end;

    return 0;

}
