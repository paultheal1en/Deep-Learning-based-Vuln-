static int img_read_header(AVFormatContext *s1)

{

    VideoDemuxData *s = s1->priv_data;

    int first_index, last_index;

    AVStream *st;

    enum AVPixelFormat pix_fmt = AV_PIX_FMT_NONE;



    s1->ctx_flags |= AVFMTCTX_NOHEADER;



    st = avformat_new_stream(s1, NULL);

    if (!st) {

        return AVERROR(ENOMEM);

    }



    if (s->pixel_format &&

        (pix_fmt = av_get_pix_fmt(s->pixel_format)) == AV_PIX_FMT_NONE) {

        av_log(s1, AV_LOG_ERROR, "No such pixel format: %s.\n",

               s->pixel_format);

        return AVERROR(EINVAL);

    }



    av_strlcpy(s->path, s1->filename, sizeof(s->path));

    s->img_number = 0;

    s->img_count  = 0;



    /* find format */

    if (s1->iformat->flags & AVFMT_NOFILE)

        s->is_pipe = 0;

    else {

        s->is_pipe       = 1;

        st->need_parsing = AVSTREAM_PARSE_FULL;

    }



    if (s->ts_from_file)

        avpriv_set_pts_info(st, 60, 1, 1);

    else

        avpriv_set_pts_info(st, 60, s->framerate.den, s->framerate.num);



    if (s->width && s->height) {

        st->codec->width  = s->width;

        st->codec->height = s->height;

    }



    if (!s->is_pipe) {

        if (s->pattern_type == PT_GLOB_SEQUENCE) {

        s->use_glob = is_glob(s->path);

        if (s->use_glob) {

            char *p = s->path, *q, *dup;

            int gerr;



            av_log(s1, AV_LOG_WARNING, "Pattern type 'glob_sequence' is deprecated: "

                   "use pattern_type 'glob' instead\n");

#if HAVE_GLOB

            dup = q = av_strdup(p);

            while (*q) {

                /* Do we have room for the next char and a \ insertion? */

                if ((p - s->path) >= (sizeof(s->path) - 2))

                  break;

                if (*q == '%' && strspn(q + 1, "%*?[]{}"))

                    ++q;

                else if (strspn(q, "\\*?[]{}"))

                    *p++ = '\\';

                *p++ = *q++;

            }

            *p = 0;

            av_free(dup);



            gerr = glob(s->path, GLOB_NOCHECK|GLOB_BRACE|GLOB_NOMAGIC, NULL, &s->globstate);

            if (gerr != 0) {

                return AVERROR(ENOENT);

            }

            first_index = 0;

            last_index = s->globstate.gl_pathc - 1;

#endif

        }

        }

        if ((s->pattern_type == PT_GLOB_SEQUENCE && !s->use_glob) || s->pattern_type == PT_SEQUENCE) {

            if (find_image_range(&first_index, &last_index, s->path,

                                 s->start_number, s->start_number_range) < 0) {

                av_log(s1, AV_LOG_ERROR,

                       "Could find no file with with path '%s' and index in the range %d-%d\n",

                       s->path, s->start_number, s->start_number + s->start_number_range - 1);

                return AVERROR(ENOENT);

            }

        } else if (s->pattern_type == PT_GLOB) {

#if HAVE_GLOB

            int gerr;

            gerr = glob(s->path, GLOB_NOCHECK|GLOB_BRACE|GLOB_NOMAGIC, NULL, &s->globstate);

            if (gerr != 0) {

                return AVERROR(ENOENT);

            }

            first_index = 0;

            last_index = s->globstate.gl_pathc - 1;

            s->use_glob = 1;

#else

            av_log(s1, AV_LOG_ERROR,

                   "Pattern type 'glob' was selected but globbing "

                   "is not supported by this libavformat build\n");

            return AVERROR(ENOSYS);

#endif

        } else if (s->pattern_type != PT_GLOB_SEQUENCE) {

            av_log(s1, AV_LOG_ERROR,

                   "Unknown value '%d' for pattern_type option\n", s->pattern_type);

            return AVERROR(EINVAL);

        }

        s->img_first  = first_index;

        s->img_last   = last_index;

        s->img_number = first_index;

        /* compute duration */

        st->start_time = 0;

        st->duration   = last_index - first_index + 1;

    }



    if (s1->video_codec_id) {

        st->codec->codec_type = AVMEDIA_TYPE_VIDEO;

        st->codec->codec_id   = s1->video_codec_id;

    } else if (s1->audio_codec_id) {

        st->codec->codec_type = AVMEDIA_TYPE_AUDIO;

        st->codec->codec_id   = s1->audio_codec_id;

    } else {

        const char *str = strrchr(s->path, '.');

        s->split_planes       = str && !av_strcasecmp(str + 1, "y");

        st->codec->codec_type = AVMEDIA_TYPE_VIDEO;

        st->codec->codec_id   = ff_guess_image2_codec(s->path);

        if (st->codec->codec_id == AV_CODEC_ID_LJPEG)

            st->codec->codec_id = AV_CODEC_ID_MJPEG;

    }

    if (st->codec->codec_type == AVMEDIA_TYPE_VIDEO &&

        pix_fmt != AV_PIX_FMT_NONE)

        st->codec->pix_fmt = pix_fmt;



    return 0;

}
