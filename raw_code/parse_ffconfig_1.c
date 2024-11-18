static int parse_ffconfig(const char *filename)

{

    FILE *f;

    char line[1024];

    char cmd[64];

    char arg[1024];

    const char *p;

    int val, errors, line_num;

    FFStream **last_stream, *stream, *redirect;

    FFStream **last_feed, *feed, *s;

    AVCodecContext audio_enc, video_enc;

    enum AVCodecID audio_id, video_id;



    f = fopen(filename, "r");

    if (!f) {

        perror(filename);

        return -1;

    }



    errors = 0;

    line_num = 0;

    first_stream = NULL;

    last_stream = &first_stream;

    first_feed = NULL;

    last_feed = &first_feed;

    stream = NULL;

    feed = NULL;

    redirect = NULL;

    audio_id = AV_CODEC_ID_NONE;

    video_id = AV_CODEC_ID_NONE;



#define ERROR(...) report_config_error(filename, line_num, &errors, __VA_ARGS__)

    for(;;) {

        if (fgets(line, sizeof(line), f) == NULL)

            break;

        line_num++;

        p = line;

        while (av_isspace(*p))

            p++;

        if (*p == '\0' || *p == '#')

            continue;



        get_arg(cmd, sizeof(cmd), &p);



        if (!av_strcasecmp(cmd, "Port")) {

            get_arg(arg, sizeof(arg), &p);

            val = atoi(arg);

            if (val < 1 || val > 65536) {

                ERROR("Invalid_port: %s\n", arg);

            }

            my_http_addr.sin_port = htons(val);

        } else if (!av_strcasecmp(cmd, "BindAddress")) {

            get_arg(arg, sizeof(arg), &p);

            if (resolve_host(&my_http_addr.sin_addr, arg) != 0) {

                ERROR("%s:%d: Invalid host/IP address: %s\n", arg);

            }

        } else if (!av_strcasecmp(cmd, "NoDaemon")) {

            // do nothing here, its the default now

        } else if (!av_strcasecmp(cmd, "RTSPPort")) {

            get_arg(arg, sizeof(arg), &p);

            val = atoi(arg);

            if (val < 1 || val > 65536) {

                ERROR("%s:%d: Invalid port: %s\n", arg);

            }

            my_rtsp_addr.sin_port = htons(atoi(arg));

        } else if (!av_strcasecmp(cmd, "RTSPBindAddress")) {

            get_arg(arg, sizeof(arg), &p);

            if (resolve_host(&my_rtsp_addr.sin_addr, arg) != 0) {

                ERROR("Invalid host/IP address: %s\n", arg);

            }

        } else if (!av_strcasecmp(cmd, "MaxHTTPConnections")) {

            get_arg(arg, sizeof(arg), &p);

            val = atoi(arg);

            if (val < 1 || val > 65536) {

                ERROR("Invalid MaxHTTPConnections: %s\n", arg);

            }

            nb_max_http_connections = val;

        } else if (!av_strcasecmp(cmd, "MaxClients")) {

            get_arg(arg, sizeof(arg), &p);

            val = atoi(arg);

            if (val < 1 || val > nb_max_http_connections) {

                ERROR("Invalid MaxClients: %s\n", arg);

            } else {

                nb_max_connections = val;

            }

        } else if (!av_strcasecmp(cmd, "MaxBandwidth")) {

            int64_t llval;

            get_arg(arg, sizeof(arg), &p);

            llval = strtoll(arg, NULL, 10);

            if (llval < 10 || llval > 10000000) {

                ERROR("Invalid MaxBandwidth: %s\n", arg);

            } else

                max_bandwidth = llval;

        } else if (!av_strcasecmp(cmd, "CustomLog")) {

            if (!ffserver_debug)

                get_arg(logfilename, sizeof(logfilename), &p);

        } else if (!av_strcasecmp(cmd, "<Feed")) {

            /*********************************************/

            /* Feed related options */

            char *q;

            if (stream || feed) {

                ERROR("Already in a tag\n");

            } else {

                feed = av_mallocz(sizeof(FFStream));

                get_arg(feed->filename, sizeof(feed->filename), &p);

                q = strrchr(feed->filename, '>');

                if (*q)

                    *q = '\0';



                for (s = first_feed; s; s = s->next) {

                    if (!strcmp(feed->filename, s->filename)) {

                        ERROR("Feed '%s' already registered\n", s->filename);

                    }

                }



                feed->fmt = av_guess_format("ffm", NULL, NULL);

                /* defaut feed file */

                snprintf(feed->feed_filename, sizeof(feed->feed_filename),

                         "/tmp/%s.ffm", feed->filename);

                feed->feed_max_size = 5 * 1024 * 1024;

                feed->is_feed = 1;

                feed->feed = feed; /* self feeding :-) */



                /* add in stream list */

                *last_stream = feed;

                last_stream = &feed->next;

                /* add in feed list */

                *last_feed = feed;

                last_feed = &feed->next_feed;

            }

        } else if (!av_strcasecmp(cmd, "Launch")) {

            if (feed) {

                int i;



                feed->child_argv = av_mallocz(64 * sizeof(char *));



                for (i = 0; i < 62; i++) {

                    get_arg(arg, sizeof(arg), &p);

                    if (!arg[0])

                        break;



                    feed->child_argv[i] = av_strdup(arg);

                }



                feed->child_argv[i] = av_asprintf("http://%s:%d/%s",

                        (my_http_addr.sin_addr.s_addr == INADDR_ANY) ? "127.0.0.1" :

                    inet_ntoa(my_http_addr.sin_addr),

                    ntohs(my_http_addr.sin_port), feed->filename);

            }

        } else if (!av_strcasecmp(cmd, "ReadOnlyFile")) {

            if (feed) {

                get_arg(feed->feed_filename, sizeof(feed->feed_filename), &p);

                feed->readonly = 1;

            } else if (stream) {

                get_arg(stream->feed_filename, sizeof(stream->feed_filename), &p);

            }

        } else if (!av_strcasecmp(cmd, "File")) {

            if (feed) {

                get_arg(feed->feed_filename, sizeof(feed->feed_filename), &p);

            } else if (stream)

                get_arg(stream->feed_filename, sizeof(stream->feed_filename), &p);

        } else if (!av_strcasecmp(cmd, "Truncate")) {

            if (feed) {

                get_arg(arg, sizeof(arg), &p);

                feed->truncate = strtod(arg, NULL);

            }

        } else if (!av_strcasecmp(cmd, "FileMaxSize")) {

            if (feed) {

                char *p1;

                double fsize;



                get_arg(arg, sizeof(arg), &p);

                p1 = arg;

                fsize = strtod(p1, &p1);

                switch(av_toupper(*p1)) {

                case 'K':

                    fsize *= 1024;

                    break;

                case 'M':

                    fsize *= 1024 * 1024;

                    break;

                case 'G':

                    fsize *= 1024 * 1024 * 1024;

                    break;

                }

                feed->feed_max_size = (int64_t)fsize;

                if (feed->feed_max_size < FFM_PACKET_SIZE*4) {

                    ERROR("Feed max file size is too small, must be at least %d\n", FFM_PACKET_SIZE*4);

                }

            }

        } else if (!av_strcasecmp(cmd, "</Feed>")) {

            if (!feed) {

                ERROR("No corresponding <Feed> for </Feed>\n");

            }

            feed = NULL;

        } else if (!av_strcasecmp(cmd, "<Stream")) {

            /*********************************************/

            /* Stream related options */

            char *q;

            if (stream || feed) {

                ERROR("Already in a tag\n");

            } else {

                FFStream *s;

                stream = av_mallocz(sizeof(FFStream));

                get_arg(stream->filename, sizeof(stream->filename), &p);

                q = strrchr(stream->filename, '>');

                if (q)

                    *q = '\0';



                for (s = first_stream; s; s = s->next) {

                    if (!strcmp(stream->filename, s->filename)) {

                        ERROR("Stream '%s' already registered\n", s->filename);

                    }

                }



                stream->fmt = ffserver_guess_format(NULL, stream->filename, NULL);

                avcodec_get_context_defaults3(&video_enc, NULL);

                avcodec_get_context_defaults3(&audio_enc, NULL);



                audio_id = AV_CODEC_ID_NONE;

                video_id = AV_CODEC_ID_NONE;

                if (stream->fmt) {

                    audio_id = stream->fmt->audio_codec;

                    video_id = stream->fmt->video_codec;

                }



                *last_stream = stream;

                last_stream = &stream->next;

            }

        } else if (!av_strcasecmp(cmd, "Feed")) {

            get_arg(arg, sizeof(arg), &p);

            if (stream) {

                FFStream *sfeed;



                sfeed = first_feed;

                while (sfeed != NULL) {

                    if (!strcmp(sfeed->filename, arg))

                        break;

                    sfeed = sfeed->next_feed;

                }

                if (!sfeed)

                    ERROR("feed '%s' not defined\n", arg);

                else

                    stream->feed = sfeed;

            }

        } else if (!av_strcasecmp(cmd, "Format")) {

            get_arg(arg, sizeof(arg), &p);

            if (stream) {

                if (!strcmp(arg, "status")) {

                    stream->stream_type = STREAM_TYPE_STATUS;

                    stream->fmt = NULL;

                } else {

                    stream->stream_type = STREAM_TYPE_LIVE;

                    /* jpeg cannot be used here, so use single frame jpeg */

                    if (!strcmp(arg, "jpeg"))

                        strcpy(arg, "mjpeg");

                    stream->fmt = ffserver_guess_format(arg, NULL, NULL);

                    if (!stream->fmt) {

                        ERROR("Unknown Format: %s\n", arg);

                    }

                }

                if (stream->fmt) {

                    audio_id = stream->fmt->audio_codec;

                    video_id = stream->fmt->video_codec;

                }

            }

        } else if (!av_strcasecmp(cmd, "InputFormat")) {

            get_arg(arg, sizeof(arg), &p);

            if (stream) {

                stream->ifmt = av_find_input_format(arg);

                if (!stream->ifmt) {

                    ERROR("Unknown input format: %s\n", arg);

                }

            }

        } else if (!av_strcasecmp(cmd, "FaviconURL")) {

            if (stream && stream->stream_type == STREAM_TYPE_STATUS) {

                get_arg(stream->feed_filename, sizeof(stream->feed_filename), &p);

            } else {

                ERROR("FaviconURL only permitted for status streams\n");

            }

        } else if (!av_strcasecmp(cmd, "Author")) {

            if (stream)

                get_arg(stream->author, sizeof(stream->author), &p);

        } else if (!av_strcasecmp(cmd, "Comment")) {

            if (stream)

                get_arg(stream->comment, sizeof(stream->comment), &p);

        } else if (!av_strcasecmp(cmd, "Copyright")) {

            if (stream)

                get_arg(stream->copyright, sizeof(stream->copyright), &p);

        } else if (!av_strcasecmp(cmd, "Title")) {

            if (stream)

                get_arg(stream->title, sizeof(stream->title), &p);

        } else if (!av_strcasecmp(cmd, "Preroll")) {

            get_arg(arg, sizeof(arg), &p);

            if (stream)

                stream->prebuffer = atof(arg) * 1000;

        } else if (!av_strcasecmp(cmd, "StartSendOnKey")) {

            if (stream)

                stream->send_on_key = 1;

        } else if (!av_strcasecmp(cmd, "AudioCodec")) {

            get_arg(arg, sizeof(arg), &p);

            audio_id = opt_codec(arg, AVMEDIA_TYPE_AUDIO);

            if (audio_id == AV_CODEC_ID_NONE) {

                ERROR("Unknown AudioCodec: %s\n", arg);

            }

        } else if (!av_strcasecmp(cmd, "VideoCodec")) {

            get_arg(arg, sizeof(arg), &p);

            video_id = opt_codec(arg, AVMEDIA_TYPE_VIDEO);

            if (video_id == AV_CODEC_ID_NONE) {

                ERROR("Unknown VideoCodec: %s\n", arg);

            }

        } else if (!av_strcasecmp(cmd, "MaxTime")) {

            get_arg(arg, sizeof(arg), &p);

            if (stream)

                stream->max_time = atof(arg) * 1000;

        } else if (!av_strcasecmp(cmd, "AudioBitRate")) {

            get_arg(arg, sizeof(arg), &p);

            if (stream)

                audio_enc.bit_rate = lrintf(atof(arg) * 1000);

        } else if (!av_strcasecmp(cmd, "AudioChannels")) {

            get_arg(arg, sizeof(arg), &p);

            if (stream)

                audio_enc.channels = atoi(arg);

        } else if (!av_strcasecmp(cmd, "AudioSampleRate")) {

            get_arg(arg, sizeof(arg), &p);

            if (stream)

                audio_enc.sample_rate = atoi(arg);

        } else if (!av_strcasecmp(cmd, "AudioQuality")) {

            get_arg(arg, sizeof(arg), &p);

            if (stream) {

//                audio_enc.quality = atof(arg) * 1000;

            }

        } else if (!av_strcasecmp(cmd, "VideoBitRateRange")) {

            if (stream) {

                int minrate, maxrate;



                get_arg(arg, sizeof(arg), &p);



                if (sscanf(arg, "%d-%d", &minrate, &maxrate) == 2) {

                    video_enc.rc_min_rate = minrate * 1000;

                    video_enc.rc_max_rate = maxrate * 1000;

                } else {

                    ERROR("Incorrect format for VideoBitRateRange -- should be <min>-<max>: %s\n", arg);

                }

            }

        } else if (!av_strcasecmp(cmd, "Debug")) {

            if (stream) {

                get_arg(arg, sizeof(arg), &p);

                video_enc.debug = strtol(arg,0,0);

            }

        } else if (!av_strcasecmp(cmd, "Strict")) {

            if (stream) {

                get_arg(arg, sizeof(arg), &p);

                video_enc.strict_std_compliance = atoi(arg);

            }

        } else if (!av_strcasecmp(cmd, "VideoBufferSize")) {

            if (stream) {

                get_arg(arg, sizeof(arg), &p);

                video_enc.rc_buffer_size = atoi(arg) * 8*1024;

            }

        } else if (!av_strcasecmp(cmd, "VideoBitRateTolerance")) {

            if (stream) {

                get_arg(arg, sizeof(arg), &p);

                video_enc.bit_rate_tolerance = atoi(arg) * 1000;

            }

        } else if (!av_strcasecmp(cmd, "VideoBitRate")) {

            get_arg(arg, sizeof(arg), &p);

            if (stream) {

                video_enc.bit_rate = atoi(arg) * 1000;

            }

        } else if (!av_strcasecmp(cmd, "VideoSize")) {

            get_arg(arg, sizeof(arg), &p);

            if (stream) {

                av_parse_video_size(&video_enc.width, &video_enc.height, arg);

                if ((video_enc.width % 16) != 0 ||

                    (video_enc.height % 16) != 0) {

                    ERROR("Image size must be a multiple of 16\n");

                }

            }

        } else if (!av_strcasecmp(cmd, "VideoFrameRate")) {

            get_arg(arg, sizeof(arg), &p);

            if (stream) {

                AVRational frame_rate;

                if (av_parse_video_rate(&frame_rate, arg) < 0) {

                    ERROR("Incorrect frame rate: %s\n", arg);

                } else {

                    video_enc.time_base.num = frame_rate.den;

                    video_enc.time_base.den = frame_rate.num;

                }

            }

        } else if (!av_strcasecmp(cmd, "PixelFormat")) {

            get_arg(arg, sizeof(arg), &p);

            if (stream) {

                video_enc.pix_fmt = av_get_pix_fmt(arg);

                if (video_enc.pix_fmt == AV_PIX_FMT_NONE) {

                    ERROR("Unknown pixel format: %s\n", arg);

                }

            }

        } else if (!av_strcasecmp(cmd, "VideoGopSize")) {

            get_arg(arg, sizeof(arg), &p);

            if (stream)

                video_enc.gop_size = atoi(arg);

        } else if (!av_strcasecmp(cmd, "VideoIntraOnly")) {

            if (stream)

                video_enc.gop_size = 1;

        } else if (!av_strcasecmp(cmd, "VideoHighQuality")) {

            if (stream)

                video_enc.mb_decision = FF_MB_DECISION_BITS;

        } else if (!av_strcasecmp(cmd, "Video4MotionVector")) {

            if (stream) {

                video_enc.mb_decision = FF_MB_DECISION_BITS; //FIXME remove

                video_enc.flags |= CODEC_FLAG_4MV;

            }

        } else if (!av_strcasecmp(cmd, "AVOptionVideo") ||

                   !av_strcasecmp(cmd, "AVOptionAudio")) {

            char arg2[1024];

            AVCodecContext *avctx;

            int type;

            get_arg(arg, sizeof(arg), &p);

            get_arg(arg2, sizeof(arg2), &p);

            if (!av_strcasecmp(cmd, "AVOptionVideo")) {

                avctx = &video_enc;

                type = AV_OPT_FLAG_VIDEO_PARAM;

            } else {

                avctx = &audio_enc;

                type = AV_OPT_FLAG_AUDIO_PARAM;

            }

            if (ffserver_opt_default(arg, arg2, avctx, type|AV_OPT_FLAG_ENCODING_PARAM)) {

                ERROR("Error setting %s option to %s %s\n", cmd, arg, arg2);

            }

        } else if (!av_strcasecmp(cmd, "AVPresetVideo") ||

                   !av_strcasecmp(cmd, "AVPresetAudio")) {

            AVCodecContext *avctx;

            int type;

            get_arg(arg, sizeof(arg), &p);

            if (!av_strcasecmp(cmd, "AVPresetVideo")) {

                avctx = &video_enc;

                video_enc.codec_id = video_id;

                type = AV_OPT_FLAG_VIDEO_PARAM;

            } else {

                avctx = &audio_enc;

                audio_enc.codec_id = audio_id;

                type = AV_OPT_FLAG_AUDIO_PARAM;

            }

            if (ffserver_opt_preset(arg, avctx, type|AV_OPT_FLAG_ENCODING_PARAM, &audio_id, &video_id)) {

                ERROR("AVPreset error: %s\n", arg);

            }

        } else if (!av_strcasecmp(cmd, "VideoTag")) {

            get_arg(arg, sizeof(arg), &p);

            if ((strlen(arg) == 4) && stream)

                video_enc.codec_tag = MKTAG(arg[0], arg[1], arg[2], arg[3]);

        } else if (!av_strcasecmp(cmd, "BitExact")) {

            if (stream)

                video_enc.flags |= CODEC_FLAG_BITEXACT;

        } else if (!av_strcasecmp(cmd, "DctFastint")) {

            if (stream)

                video_enc.dct_algo  = FF_DCT_FASTINT;

        } else if (!av_strcasecmp(cmd, "IdctSimple")) {

            if (stream)

                video_enc.idct_algo = FF_IDCT_SIMPLE;

        } else if (!av_strcasecmp(cmd, "Qscale")) {

            get_arg(arg, sizeof(arg), &p);

            if (stream) {

                video_enc.flags |= CODEC_FLAG_QSCALE;

                video_enc.global_quality = FF_QP2LAMBDA * atoi(arg);

            }

        } else if (!av_strcasecmp(cmd, "VideoQDiff")) {

            get_arg(arg, sizeof(arg), &p);

            if (stream) {

                video_enc.max_qdiff = atoi(arg);

                if (video_enc.max_qdiff < 1 || video_enc.max_qdiff > 31) {

                    ERROR("VideoQDiff out of range\n");

                }

            }

        } else if (!av_strcasecmp(cmd, "VideoQMax")) {

            get_arg(arg, sizeof(arg), &p);

            if (stream) {

                video_enc.qmax = atoi(arg);

                if (video_enc.qmax < 1 || video_enc.qmax > 31) {

                    ERROR("VideoQMax out of range\n");

                }

            }

        } else if (!av_strcasecmp(cmd, "VideoQMin")) {

            get_arg(arg, sizeof(arg), &p);

            if (stream) {

                video_enc.qmin = atoi(arg);

                if (video_enc.qmin < 1 || video_enc.qmin > 31) {

                    ERROR("VideoQMin out of range\n");

                }

            }

        } else if (!av_strcasecmp(cmd, "LumiMask")) {

            get_arg(arg, sizeof(arg), &p);

            if (stream)

                video_enc.lumi_masking = atof(arg);

        } else if (!av_strcasecmp(cmd, "DarkMask")) {

            get_arg(arg, sizeof(arg), &p);

            if (stream)

                video_enc.dark_masking = atof(arg);

        } else if (!av_strcasecmp(cmd, "NoVideo")) {

            video_id = AV_CODEC_ID_NONE;

        } else if (!av_strcasecmp(cmd, "NoAudio")) {

            audio_id = AV_CODEC_ID_NONE;

        } else if (!av_strcasecmp(cmd, "ACL")) {

            parse_acl_row(stream, feed, NULL, p, filename, line_num);

        } else if (!av_strcasecmp(cmd, "DynamicACL")) {

            if (stream) {

                get_arg(stream->dynamic_acl, sizeof(stream->dynamic_acl), &p);

            }

        } else if (!av_strcasecmp(cmd, "RTSPOption")) {

            get_arg(arg, sizeof(arg), &p);

            if (stream) {

                av_freep(&stream->rtsp_option);

                stream->rtsp_option = av_strdup(arg);

            }

        } else if (!av_strcasecmp(cmd, "MulticastAddress")) {

            get_arg(arg, sizeof(arg), &p);

            if (stream) {

                if (resolve_host(&stream->multicast_ip, arg) != 0) {

                    ERROR("Invalid host/IP address: %s\n", arg);

                }

                stream->is_multicast = 1;

                stream->loop = 1; /* default is looping */

            }

        } else if (!av_strcasecmp(cmd, "MulticastPort")) {

            get_arg(arg, sizeof(arg), &p);

            if (stream)

                stream->multicast_port = atoi(arg);

        } else if (!av_strcasecmp(cmd, "MulticastTTL")) {

            get_arg(arg, sizeof(arg), &p);

            if (stream)

                stream->multicast_ttl = atoi(arg);

        } else if (!av_strcasecmp(cmd, "NoLoop")) {

            if (stream)

                stream->loop = 0;

        } else if (!av_strcasecmp(cmd, "</Stream>")) {

            if (!stream) {

                ERROR("No corresponding <Stream> for </Stream>\n");

            } else {

                if (stream->feed && stream->fmt && strcmp(stream->fmt->name, "ffm") != 0) {

                    if (audio_id != AV_CODEC_ID_NONE) {

                        audio_enc.codec_type = AVMEDIA_TYPE_AUDIO;

                        audio_enc.codec_id = audio_id;

                        add_codec(stream, &audio_enc);

                    }

                    if (video_id != AV_CODEC_ID_NONE) {

                        video_enc.codec_type = AVMEDIA_TYPE_VIDEO;

                        video_enc.codec_id = video_id;

                        add_codec(stream, &video_enc);

                    }

                }

                stream = NULL;

            }

        } else if (!av_strcasecmp(cmd, "<Redirect")) {

            /*********************************************/

            char *q;

            if (stream || feed || redirect) {

                ERROR("Already in a tag\n");

            } else {

                redirect = av_mallocz(sizeof(FFStream));

                *last_stream = redirect;

                last_stream = &redirect->next;



                get_arg(redirect->filename, sizeof(redirect->filename), &p);

                q = strrchr(redirect->filename, '>');

                if (*q)

                    *q = '\0';

                redirect->stream_type = STREAM_TYPE_REDIRECT;

            }

        } else if (!av_strcasecmp(cmd, "URL")) {

            if (redirect)

                get_arg(redirect->feed_filename, sizeof(redirect->feed_filename), &p);

        } else if (!av_strcasecmp(cmd, "</Redirect>")) {

            if (!redirect) {

                ERROR("No corresponding <Redirect> for </Redirect>\n");

            } else {

                if (!redirect->feed_filename[0]) {

                    ERROR("No URL found for <Redirect>\n");

                }

                redirect = NULL;

            }

        } else if (!av_strcasecmp(cmd, "LoadModule")) {

            ERROR("Loadable modules no longer supported\n");

        } else {

            ERROR("Incorrect keyword: '%s'\n", cmd);

        }

    }

#undef ERROR



    fclose(f);

    if (errors)

        return -1;

    else

        return 0;

}
