static int nut_write_header(AVFormatContext *s)

{

    NUTContext *nut = s->priv_data;

    AVIOContext *bc = s->pb;

    int i, j, ret;



    nut->avf = s;



    nut->version = FFMAX(NUT_STABLE_VERSION, 3 + !!nut->flags);

    if (nut->flags && s->strict_std_compliance > FF_COMPLIANCE_EXPERIMENTAL) {

        av_log(s, AV_LOG_ERROR,

               "The additional syncpoint modes require version %d, "

               "that is currently not finalized, "

               "please set -f_strict experimental in order to enable it.\n",

               nut->version);

        return AVERROR_EXPERIMENTAL;

    }



    nut->stream   = av_calloc(s->nb_streams,  sizeof(*nut->stream ));

    nut->chapter  = av_calloc(s->nb_chapters, sizeof(*nut->chapter));

    nut->time_base= av_calloc(s->nb_streams +

                              s->nb_chapters, sizeof(*nut->time_base));

    if (!nut->stream || !nut->chapter || !nut->time_base) {

        av_freep(&nut->stream);

        av_freep(&nut->chapter);

        av_freep(&nut->time_base);

        return AVERROR(ENOMEM);

    }



    for (i = 0; i < s->nb_streams; i++) {

        AVStream *st = s->streams[i];

        int ssize;

        AVRational time_base;

        ff_parse_specific_params(st->codec, &time_base.den, &ssize, &time_base.num);



        if (st->codec->codec_type == AVMEDIA_TYPE_AUDIO && st->codec->sample_rate) {

            time_base = (AVRational) {1, st->codec->sample_rate};

        } else {

            time_base = ff_choose_timebase(s, st, 48000);

        }



        avpriv_set_pts_info(st, 64, time_base.num, time_base.den);



        for (j = 0; j < nut->time_base_count; j++)

            if (!memcmp(&time_base, &nut->time_base[j], sizeof(AVRational))) {

                break;

            }

        nut->time_base[j]        = time_base;

        nut->stream[i].time_base = &nut->time_base[j];

        if (j == nut->time_base_count)

            nut->time_base_count++;



        if (INT64_C(1000) * time_base.num >= time_base.den)

            nut->stream[i].msb_pts_shift = 7;

        else

            nut->stream[i].msb_pts_shift = 14;

        nut->stream[i].max_pts_distance =

            FFMAX(time_base.den, time_base.num) / time_base.num;

    }



    for (i = 0; i < s->nb_chapters; i++) {

        AVChapter *ch = s->chapters[i];



        for (j = 0; j < nut->time_base_count; j++)

            if (!memcmp(&ch->time_base, &nut->time_base[j], sizeof(AVRational)))

                break;



        nut->time_base[j]         = ch->time_base;

        nut->chapter[i].time_base = &nut->time_base[j];

        if (j == nut->time_base_count)

            nut->time_base_count++;

    }



    nut->max_distance = MAX_DISTANCE;

    build_elision_headers(s);

    build_frame_code(s);

    av_assert0(nut->frame_code['N'].flags == FLAG_INVALID);



    avio_write(bc, ID_STRING, strlen(ID_STRING));

    avio_w8(bc, 0);



    if ((ret = write_headers(s, bc)) < 0)

        return ret;



    if (s->avoid_negative_ts < 0)

        s->avoid_negative_ts = 1;



    avio_flush(bc);



    return 0;

}
