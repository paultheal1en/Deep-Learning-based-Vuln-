static int sdp_parse_rtpmap(AVFormatContext *s,

                            AVStream *st, RTSPStream *rtsp_st,

                            int payload_type, const char *p)

{

    AVCodecContext *codec = st->codec;

    char buf[256];

    int i;

    AVCodec *c;

    const char *c_name;



    /* See if we can handle this kind of payload.

     * The space should normally not be there but some Real streams or

     * particular servers ("RealServer Version 6.1.3.970", see issue 1658)

     * have a trailing space. */

    get_word_sep(buf, sizeof(buf), "/ ", &p);

    if (payload_type < RTP_PT_PRIVATE) {

        /* We are in a standard case

         * (from http://www.iana.org/assignments/rtp-parameters). */

        codec->codec_id = ff_rtp_codec_id(buf, codec->codec_type);

    }



    if (codec->codec_id == AV_CODEC_ID_NONE) {

        RTPDynamicProtocolHandler *handler =

            ff_rtp_handler_find_by_name(buf, codec->codec_type);

        init_rtp_handler(handler, rtsp_st, st);

        /* If no dynamic handler was found, check with the list of standard

         * allocated types, if such a stream for some reason happens to

         * use a private payload type. This isn't handled in rtpdec.c, since

         * the format name from the rtpmap line never is passed into rtpdec. */

        if (!rtsp_st->dynamic_handler)

            codec->codec_id = ff_rtp_codec_id(buf, codec->codec_type);

    }



    c = avcodec_find_decoder(codec->codec_id);

    if (c && c->name)

        c_name = c->name;

    else

        c_name = "(null)";



    get_word_sep(buf, sizeof(buf), "/", &p);

    i = atoi(buf);

    switch (codec->codec_type) {

    case AVMEDIA_TYPE_AUDIO:

        av_log(s, AV_LOG_DEBUG, "audio codec set to: %s\n", c_name);

        codec->sample_rate = RTSP_DEFAULT_AUDIO_SAMPLERATE;

        codec->channels = RTSP_DEFAULT_NB_AUDIO_CHANNELS;

        if (i > 0) {

            codec->sample_rate = i;

            avpriv_set_pts_info(st, 32, 1, codec->sample_rate);

            get_word_sep(buf, sizeof(buf), "/", &p);

            i = atoi(buf);

            if (i > 0)

                codec->channels = i;

        }

        av_log(s, AV_LOG_DEBUG, "audio samplerate set to: %i\n",

               codec->sample_rate);

        av_log(s, AV_LOG_DEBUG, "audio channels set to: %i\n",

               codec->channels);

        break;

    case AVMEDIA_TYPE_VIDEO:

        av_log(s, AV_LOG_DEBUG, "video codec set to: %s\n", c_name);

        if (i > 0)

            avpriv_set_pts_info(st, 32, 1, i);

        break;

    default:

        break;

    }

    if (rtsp_st->dynamic_handler && rtsp_st->dynamic_handler->init)

        rtsp_st->dynamic_handler->init(s, st->index,

                                       rtsp_st->dynamic_protocol_context);

    return 0;

}
