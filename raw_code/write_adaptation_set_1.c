static int write_adaptation_set(AVFormatContext *s, int as_index)

{

    WebMDashMuxContext *w = s->priv_data;

    AdaptationSet *as = &w->as[as_index];

    AVCodecContext *codec = s->streams[as->streams[0]]->codec;

    AVDictionaryEntry *lang;

    int i;

    static const char boolean[2][6] = { "false", "true" };

    int subsegmentStartsWithSAP = 1;



    // Width, Height and Sample Rate will go in the AdaptationSet tag if they

    // are the same for all contained Representations. otherwise, they will go

    // on their respective Representation tag. For live streams, they always go

    // in the Representation tag.

    int width_in_as = 1, height_in_as = 1, sample_rate_in_as = 1;

    if (codec->codec_type == AVMEDIA_TYPE_VIDEO) {

      width_in_as = !w->is_live && check_matching_width(s, as);

      height_in_as = !w->is_live && check_matching_height(s, as);

    } else {

      sample_rate_in_as = !w->is_live && check_matching_sample_rate(s, as);

    }



    avio_printf(s->pb, "<AdaptationSet id=\"%s\"", as->id);

    avio_printf(s->pb, " mimeType=\"%s/webm\"",

                codec->codec_type == AVMEDIA_TYPE_VIDEO ? "video" : "audio");

    avio_printf(s->pb, " codecs=\"%s\"", get_codec_name(codec->codec_id));



    lang = av_dict_get(s->streams[as->streams[0]]->metadata, "language", NULL, 0);

    if (lang) avio_printf(s->pb, " lang=\"%s\"", lang->value);



    if (codec->codec_type == AVMEDIA_TYPE_VIDEO && width_in_as)

        avio_printf(s->pb, " width=\"%d\"", codec->width);

    if (codec->codec_type == AVMEDIA_TYPE_VIDEO && height_in_as)

        avio_printf(s->pb, " height=\"%d\"", codec->height);

    if (codec->codec_type == AVMEDIA_TYPE_AUDIO && sample_rate_in_as)

        avio_printf(s->pb, " audioSamplingRate=\"%d\"", codec->sample_rate);



    avio_printf(s->pb, " bitstreamSwitching=\"%s\"",

                boolean[bitstream_switching(s, as)]);

    avio_printf(s->pb, " subsegmentAlignment=\"%s\"",

                boolean[w->is_live || subsegment_alignment(s, as)]);



    for (i = 0; i < as->nb_streams; i++) {

        AVDictionaryEntry *kf = av_dict_get(s->streams[as->streams[i]]->metadata,

                                            CLUSTER_KEYFRAME, NULL, 0);

        if (!w->is_live && (!kf || !strncmp(kf->value, "0", 1))) subsegmentStartsWithSAP = 0;

    }

    avio_printf(s->pb, " subsegmentStartsWithSAP=\"%d\"", subsegmentStartsWithSAP);

    avio_printf(s->pb, ">\n");



    if (w->is_live) {

        AVDictionaryEntry *filename =

            av_dict_get(s->streams[as->streams[0]]->metadata, FILENAME, NULL, 0);

        char *initialization_pattern = NULL;

        char *media_pattern = NULL;

        int ret = parse_filename(filename->value, NULL, &initialization_pattern,

                                 &media_pattern);

        if (ret) return ret;

        avio_printf(s->pb, "<ContentComponent id=\"1\" type=\"%s\"/>\n",

                    codec->codec_type == AVMEDIA_TYPE_VIDEO ? "video" : "audio");

        avio_printf(s->pb, "<SegmentTemplate");

        avio_printf(s->pb, " timescale=\"1000\"");

        avio_printf(s->pb, " duration=\"%d\"", w->chunk_duration);

        avio_printf(s->pb, " media=\"%s\"", media_pattern);

        avio_printf(s->pb, " startNumber=\"%d\"", w->chunk_start_index);

        avio_printf(s->pb, " initialization=\"%s\"", initialization_pattern);

        avio_printf(s->pb, "/>\n");

        av_free(initialization_pattern);

        av_free(media_pattern);

    }



    for (i = 0; i < as->nb_streams; i++) {

        char *representation_id = NULL;

        int ret;

        if (w->is_live) {

            AVDictionaryEntry *filename =

                av_dict_get(s->streams[as->streams[i]]->metadata, FILENAME, NULL, 0);

            if (!filename ||

                (ret = parse_filename(filename->value, &representation_id, NULL, NULL))) {

                return ret;

            }

        } else {

            representation_id = av_asprintf("%d", w->representation_id++);

            if (!representation_id) return AVERROR(ENOMEM);

        }

        ret = write_representation(s, s->streams[as->streams[i]],

                                   representation_id, !width_in_as,

                                   !height_in_as, !sample_rate_in_as);

        if (ret) return ret;

        av_free(representation_id);

    }

    avio_printf(s->pb, "</AdaptationSet>\n");

    return 0;

}
