static void add_codec(FFServerStream *stream, AVCodecContext *av,

                      FFServerConfig *config)

{

    AVStream *st;

    AVDictionary **opts, *recommended = NULL;

    char *enc_config;



    if(stream->nb_streams >= FF_ARRAY_ELEMS(stream->streams))

        return;



    opts = av->codec_type == AVMEDIA_TYPE_AUDIO ?

           &config->audio_opts : &config->video_opts;

    av_dict_copy(&recommended, *opts, 0);

    av_opt_set_dict2(av->priv_data, opts, AV_OPT_SEARCH_CHILDREN);

    av_opt_set_dict2(av, opts, AV_OPT_SEARCH_CHILDREN);

    if (av_dict_count(*opts))

        av_log(NULL, AV_LOG_WARNING,

               "Something is wrong, %d options are not set!\n", av_dict_count(*opts));



    if (config->stream_use_defaults) {

    //TODO: reident

    /* compute default parameters */

    switch(av->codec_type) {

    case AVMEDIA_TYPE_AUDIO:

        if (av->bit_rate == 0) {

            av->bit_rate = 64000;

            av_dict_set_int(&recommended, "ab", av->bit_rate, 0);

        }

        if (av->sample_rate == 0) {

            av->sample_rate = 22050;

            av_dict_set_int(&recommended, "ar", av->sample_rate, 0);

        }

        if (av->channels == 0) {

            av->channels = 1;

            av_dict_set_int(&recommended, "ac", av->channels, 0);

        }

        break;

    case AVMEDIA_TYPE_VIDEO:

        if (av->bit_rate == 0) {

            av->bit_rate = 64000;

            av_dict_set_int(&recommended, "b", av->bit_rate, 0);

        }

        if (av->time_base.num == 0){

            av->time_base.den = 5;

            av->time_base.num = 1;

            av_dict_set(&recommended, "time_base", "1/5", 0);

        }

        if (av->width == 0 || av->height == 0) {

            av->width = 160;

            av->height = 128;

            av_dict_set(&recommended, "video_size", "160x128", 0);

        }

        /* Bitrate tolerance is less for streaming */

        if (av->bit_rate_tolerance == 0) {

            av->bit_rate_tolerance = FFMAX(av->bit_rate / 4,

                      (int64_t)av->bit_rate*av->time_base.num/av->time_base.den);

            av_dict_set_int(&recommended, "bt", av->bit_rate_tolerance, 0);

        }



        if (!av->rc_eq) {

            av->rc_eq = av_strdup("tex^qComp");

            av_dict_set(&recommended, "rc_eq", "tex^qComp", 0);

        }

        if (!av->rc_max_rate) {

            av->rc_max_rate = av->bit_rate * 2;

            av_dict_set_int(&recommended, "maxrate", av->rc_max_rate, 0);

        }



        if (av->rc_max_rate && !av->rc_buffer_size) {

            av->rc_buffer_size = av->rc_max_rate;

            av_dict_set_int(&recommended, "bufsize", av->rc_buffer_size, 0);

        }

        break;

    default:

        abort();

    }

    } else {

        switch(av->codec_type) {

        case AVMEDIA_TYPE_AUDIO:

            if (av->bit_rate == 0)

                report_config_error(config->filename, config->line_num, AV_LOG_ERROR,

                                    &config->errors, "audio bit rate is not set\n");

            if (av->sample_rate == 0)

                report_config_error(config->filename, config->line_num, AV_LOG_ERROR,

                                    &config->errors, "audio sample rate is not set\n");

            break;

        case AVMEDIA_TYPE_VIDEO:

            if (av->width == 0 || av->height == 0)

                report_config_error(config->filename, config->line_num, AV_LOG_ERROR,

                                    &config->errors, "video size is not set\n");

            break;

        default:

            av_assert0(0);

        }

    }



    st = av_mallocz(sizeof(AVStream));

    if (!st)

        return;

    av_dict_get_string(recommended, &enc_config, '=', ',');

    av_dict_free(&recommended);

    av_stream_set_recommended_encoder_configuration(st, enc_config);

    st->codec = av;

    stream->streams[stream->nb_streams++] = st;

}
