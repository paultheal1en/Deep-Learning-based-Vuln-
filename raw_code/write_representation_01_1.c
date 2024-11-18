static int write_representation(AVFormatContext *s, AVStream *stream, char *id,

                                int output_width, int output_height,

                                int output_sample_rate) {

    WebMDashMuxContext *w = s->priv_data;

    AVDictionaryEntry *irange = av_dict_get(stream->metadata, INITIALIZATION_RANGE, NULL, 0);

    AVDictionaryEntry *cues_start = av_dict_get(stream->metadata, CUES_START, NULL, 0);

    AVDictionaryEntry *cues_end = av_dict_get(stream->metadata, CUES_END, NULL, 0);

    AVDictionaryEntry *filename = av_dict_get(stream->metadata, FILENAME, NULL, 0);

    AVDictionaryEntry *bandwidth = av_dict_get(stream->metadata, BANDWIDTH, NULL, 0);

    if ((w->is_live && (!filename)) ||

        (!w->is_live && (!irange || !cues_start || !cues_end || !filename || !bandwidth))) {

        return -1;

    }

    avio_printf(s->pb, "<Representation id=\"%s\"", id);

    // FIXME: For live, This should be obtained from the input file or as an AVOption.

    avio_printf(s->pb, " bandwidth=\"%s\"",

                w->is_live ? (stream->codec->codec_type == AVMEDIA_TYPE_AUDIO ? "128000" : "1000000") : bandwidth->value);

    if (stream->codec->codec_type == AVMEDIA_TYPE_VIDEO && output_width)

        avio_printf(s->pb, " width=\"%d\"", stream->codec->width);

    if (stream->codec->codec_type == AVMEDIA_TYPE_VIDEO && output_height)

        avio_printf(s->pb, " height=\"%d\"", stream->codec->height);

    if (stream->codec->codec_type = AVMEDIA_TYPE_AUDIO && output_sample_rate)

        avio_printf(s->pb, " audioSamplingRate=\"%d\"", stream->codec->sample_rate);

    if (w->is_live) {

        // For live streams, Codec and Mime Type always go in the Representation tag.

        avio_printf(s->pb, " codecs=\"%s\"", get_codec_name(stream->codec->codec_id));

        avio_printf(s->pb, " mimeType=\"%s/webm\"",

                    stream->codec->codec_type == AVMEDIA_TYPE_VIDEO ? "video" : "audio");

        // For live streams, subsegments always start with key frames. So this

        // is always 1.

        avio_printf(s->pb, " startsWithSAP=\"1\"");

        avio_printf(s->pb, ">");

    } else {

        avio_printf(s->pb, ">\n");

        avio_printf(s->pb, "<BaseURL>%s</BaseURL>\n", filename->value);

        avio_printf(s->pb, "<SegmentBase\n");

        avio_printf(s->pb, "  indexRange=\"%s-%s\">\n", cues_start->value, cues_end->value);

        avio_printf(s->pb, "<Initialization\n");

        avio_printf(s->pb, "  range=\"0-%s\" />\n", irange->value);

        avio_printf(s->pb, "</SegmentBase>\n");

    }

    avio_printf(s->pb, "</Representation>\n");

    return 0;

}
