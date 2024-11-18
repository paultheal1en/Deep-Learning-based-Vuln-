enum AVCodecID av_guess_codec(AVOutputFormat *fmt, const char *short_name,

                              const char *filename, const char *mime_type,

                              enum AVMediaType type)

{

    if (av_match_name("segment", fmt->name) || av_match_name("ssegment", fmt->name)) {

        fmt = av_guess_format(NULL, filename, NULL);

    }



    if (type == AVMEDIA_TYPE_VIDEO) {

        enum AVCodecID codec_id = AV_CODEC_ID_NONE;



#if CONFIG_IMAGE2_MUXER

        if (!strcmp(fmt->name, "image2") || !strcmp(fmt->name, "image2pipe")) {

            codec_id = ff_guess_image2_codec(filename);

        }

#endif

        if (codec_id == AV_CODEC_ID_NONE)

            codec_id = fmt->video_codec;

        return codec_id;

    } else if (type == AVMEDIA_TYPE_AUDIO)

        return fmt->audio_codec;

    else if (type == AVMEDIA_TYPE_SUBTITLE)

        return fmt->subtitle_codec;

    else

        return AV_CODEC_ID_NONE;

}
