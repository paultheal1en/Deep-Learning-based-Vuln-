static int mov_create_chapter_track(AVFormatContext *s, int tracknum)

{

    MOVMuxContext *mov = s->priv_data;

    MOVTrack *track = &mov->tracks[tracknum];

    AVPacket pkt = { .stream_index = tracknum, .flags = AV_PKT_FLAG_KEY };

    int i, len;

    // These properties are required to make QT recognize the chapter track

    uint8_t chapter_properties[43] = { 0, 0, 0, 0, 0, 0, 0, 1, };



    track->mode = mov->mode;

    track->tag = MKTAG('t','e','x','t');

    track->timescale = MOV_TIMESCALE;

    track->enc = avcodec_alloc_context3(NULL);

    if (!track->enc)


    track->enc->codec_type = AVMEDIA_TYPE_SUBTITLE;

    track->enc->extradata = av_malloc(sizeof(chapter_properties));

    if (track->enc->extradata == NULL)


    track->enc->extradata_size = sizeof(chapter_properties);

    memcpy(track->enc->extradata, chapter_properties, sizeof(chapter_properties));



    for (i = 0; i < s->nb_chapters; i++) {

        AVChapter *c = s->chapters[i];

        AVDictionaryEntry *t;



        int64_t end = av_rescale_q(c->end, c->time_base, (AVRational){1,MOV_TIMESCALE});

        pkt.pts = pkt.dts = av_rescale_q(c->start, c->time_base, (AVRational){1,MOV_TIMESCALE});

        pkt.duration = end - pkt.dts;



        if ((t = av_dict_get(c->metadata, "title", NULL, 0))) {

            len      = strlen(t->value);

            pkt.size = len + 2;

            pkt.data = av_malloc(pkt.size);



            AV_WB16(pkt.data, len);

            memcpy(pkt.data + 2, t->value, len);

            ff_mov_write_packet(s, &pkt);

            av_freep(&pkt.data);

        }

    }



    return 0;

}