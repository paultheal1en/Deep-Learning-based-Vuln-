ff_rm_read_mdpr_codecdata (AVFormatContext *s, AVIOContext *pb,

                           AVStream *st, RMStream *rst, int codec_data_size)

{

    unsigned int v;

    int size;

    int64_t codec_pos;

    int ret;



    avpriv_set_pts_info(st, 64, 1, 1000);

    codec_pos = avio_tell(pb);

    v = avio_rb32(pb);

    if (v == MKTAG(0xfd, 'a', 'r', '.')) {

        /* ra type header */

        if (rm_read_audio_stream_info(s, pb, st, rst, 0))

            return -1;

    } else if (v == MKBETAG('L', 'S', 'D', ':')) {

        avio_seek(pb, -4, SEEK_CUR);

        if ((ret = rm_read_extradata(pb, st->codec, codec_data_size)) < 0)

            return ret;



        st->codec->codec_type = AVMEDIA_TYPE_AUDIO;

        st->codec->codec_tag  = AV_RL32(st->codec->extradata);

        st->codec->codec_id   = ff_codec_get_id(ff_rm_codec_tags,

                                                st->codec->codec_tag);

    } else {

        int fps;

        if (avio_rl32(pb) != MKTAG('V', 'I', 'D', 'O')) {

        fail1:

            av_log(st->codec, AV_LOG_ERROR, "Unsupported video codec\n");

            goto skip;

        }

        st->codec->codec_tag = avio_rl32(pb);

        st->codec->codec_id  = ff_codec_get_id(ff_rm_codec_tags,

                                               st->codec->codec_tag);

//        av_log(s, AV_LOG_DEBUG, "%X %X\n", st->codec->codec_tag, MKTAG('R', 'V', '2', '0'));

        if (st->codec->codec_id == CODEC_ID_NONE)

            goto fail1;

        st->codec->width  = avio_rb16(pb);

        st->codec->height = avio_rb16(pb);

        avio_skip(pb, 2); // looks like bits per sample

        avio_skip(pb, 4); // always zero?

        st->codec->codec_type = AVMEDIA_TYPE_VIDEO;

        st->need_parsing = AVSTREAM_PARSE_TIMESTAMPS;

        fps = avio_rb32(pb);



        if ((ret = rm_read_extradata(pb, st->codec, codec_data_size - (avio_tell(pb) - codec_pos))) < 0)

            return ret;



        av_reduce(&st->r_frame_rate.den, &st->r_frame_rate.num,

                  0x10000, fps, (1 << 30) - 1);

        st->avg_frame_rate = st->r_frame_rate;

    }



skip:

    /* skip codec info */

    size = avio_tell(pb) - codec_pos;

    avio_skip(pb, codec_data_size - size);



    return 0;

}
