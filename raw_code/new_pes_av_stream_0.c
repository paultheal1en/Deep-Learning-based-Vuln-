static AVStream *new_pes_av_stream(PESContext *pes, uint32_t prog_reg_desc, uint32_t code)

{

    AVStream *st = av_new_stream(pes->stream, pes->pid);



    if (!st)

        return NULL;



    av_set_pts_info(st, 33, 1, 90000);

    st->priv_data = pes;

    st->codec->codec_type = CODEC_TYPE_DATA;

    st->codec->codec_id   = CODEC_ID_NONE;

    st->need_parsing = AVSTREAM_PARSE_FULL;

    pes->st = st;



    dprintf(pes->stream, "stream_type=%x pid=%x prog_reg_desc=%.4s\n",

            pes->stream_type, pes->pid, (char*)&prog_reg_desc);



    st->codec->codec_tag = pes->stream_type;



    mpegts_find_stream_type(st, pes->stream_type, ISO_types);

    if (prog_reg_desc == AV_RL32("HDMV") &&

        st->codec->codec_id == CODEC_ID_NONE) {

        mpegts_find_stream_type(st, pes->stream_type, HDMV_types);

        if (pes->stream_type == 0x83) {

            // HDMV TrueHD streams also contain an AC3 coded version of the

            // audio track - add a second stream for this

            AVStream *sub_st;

            // priv_data cannot be shared between streams

            PESContext *sub_pes = av_malloc(sizeof(*sub_pes));

            if (!sub_pes)

                return NULL;

            memcpy(sub_pes, pes, sizeof(*sub_pes));



            sub_st = av_new_stream(pes->stream, pes->pid);

            if (!sub_st) {

                av_free(sub_pes);

                return NULL;

            }



            av_set_pts_info(sub_st, 33, 1, 90000);

            sub_st->priv_data = sub_pes;

            sub_st->codec->codec_type = CODEC_TYPE_AUDIO;

            sub_st->codec->codec_id   = CODEC_ID_AC3;

            sub_st->need_parsing = AVSTREAM_PARSE_FULL;

            sub_pes->sub_st = pes->sub_st = sub_st;

        }

    }

    if (st->codec->codec_id == CODEC_ID_NONE)

        mpegts_find_stream_type(st, pes->stream_type, MISC_types);



    /* stream was not present in PMT, guess based on PES start code */

    if (st->codec->codec_id == CODEC_ID_NONE) {

        if (code >= 0x1c0 && code <= 0x1df) {

            st->codec->codec_type = CODEC_TYPE_AUDIO;

            st->codec->codec_id = CODEC_ID_MP2;

        } else if (code == 0x1bd) {

            st->codec->codec_type = CODEC_TYPE_AUDIO;

            st->codec->codec_id = CODEC_ID_AC3;

        }

    }



    return st;

}
