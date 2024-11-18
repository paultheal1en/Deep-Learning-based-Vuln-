static int seg_write_header(AVFormatContext *s)

{

    SegmentContext *seg = s->priv_data;

    AVFormatContext *oc = NULL;

    AVDictionary *options = NULL;

    int ret;



    seg->segment_count = 0;

    if (!seg->write_header_trailer)

        seg->individual_header_trailer = 0;



    if (!!seg->time_str + !!seg->times_str + !!seg->frames_str > 1) {

        av_log(s, AV_LOG_ERROR,

               "segment_time, segment_times, and segment_frames options "

               "are mutually exclusive, select just one of them\n");

        return AVERROR(EINVAL);

    }



    if (seg->times_str) {

        if ((ret = parse_times(s, &seg->times, &seg->nb_times, seg->times_str)) < 0)

            return ret;

    } else if (seg->frames_str) {

        if ((ret = parse_frames(s, &seg->frames, &seg->nb_frames, seg->frames_str)) < 0)

            return ret;

    } else {

        /* set default value if not specified */

        if (!seg->time_str)

            seg->time_str = av_strdup("2");

        if ((ret = av_parse_time(&seg->time, seg->time_str, 1)) < 0) {

            av_log(s, AV_LOG_ERROR,

                   "Invalid time duration specification '%s' for segment_time option\n",

                   seg->time_str);

            return ret;

        }

    }



    if (seg->format_options_str) {

        ret = av_dict_parse_string(&seg->format_options, seg->format_options_str, "=", ":", 0);

        if (ret < 0) {

            av_log(s, AV_LOG_ERROR, "Could not parse format options list '%s'\n",

                   seg->format_options_str);

            goto fail;

        }

    }



    if (seg->list) {

        if (seg->list_type == LIST_TYPE_UNDEFINED) {

            if      (av_match_ext(seg->list, "csv" )) seg->list_type = LIST_TYPE_CSV;

            else if (av_match_ext(seg->list, "ext" )) seg->list_type = LIST_TYPE_EXT;

            else if (av_match_ext(seg->list, "m3u8")) seg->list_type = LIST_TYPE_M3U8;

            else if (av_match_ext(seg->list, "ffcat,ffconcat")) seg->list_type = LIST_TYPE_FFCONCAT;

            else                                      seg->list_type = LIST_TYPE_FLAT;

        }

        if ((ret = segment_list_open(s)) < 0)

            goto fail;

    }

    if (seg->list_type == LIST_TYPE_EXT)

        av_log(s, AV_LOG_WARNING, "'ext' list type option is deprecated in favor of 'csv'\n");



    if ((ret = select_reference_stream(s)) < 0)

        goto fail;

    av_log(s, AV_LOG_VERBOSE, "Selected stream id:%d type:%s\n",

           seg->reference_stream_index,

           av_get_media_type_string(s->streams[seg->reference_stream_index]->codec->codec_type));



    seg->oformat = av_guess_format(seg->format, s->filename, NULL);



    if (!seg->oformat) {

        ret = AVERROR_MUXER_NOT_FOUND;

        goto fail;

    }

    if (seg->oformat->flags & AVFMT_NOFILE) {

        av_log(s, AV_LOG_ERROR, "format %s not supported.\n",

               seg->oformat->name);

        ret = AVERROR(EINVAL);

        goto fail;

    }



    if ((ret = segment_mux_init(s)) < 0)

        goto fail;

    oc = seg->avf;



    if ((ret = set_segment_filename(s)) < 0)

        goto fail;



    if (seg->write_header_trailer) {

        if ((ret = avio_open2(&oc->pb, oc->filename, AVIO_FLAG_WRITE,

                              &s->interrupt_callback, NULL)) < 0) {

            av_log(s, AV_LOG_ERROR, "Failed to open segment '%s'\n", oc->filename);

            goto fail;

        }

    } else {

        if ((ret = open_null_ctx(&oc->pb)) < 0)

            goto fail;

    }



    av_dict_copy(&options, seg->format_options, 0);

    ret = avformat_write_header(oc, &options);

    if (av_dict_count(options)) {

        av_log(s, AV_LOG_ERROR,

               "Some of the provided format options in '%s' are not recognized\n", seg->format_options_str);

    }

    av_dict_free(&options);

    if (ret < 0) {

        avio_close(oc->pb);

        goto fail;

    }

    seg->segment_frame_count = 0;



    if (oc->avoid_negative_ts > 0 && s->avoid_negative_ts < 0)

        s->avoid_negative_ts = 1;



    if (!seg->write_header_trailer) {

        close_null_ctx(oc->pb);

        if ((ret = avio_open2(&oc->pb, oc->filename, AVIO_FLAG_WRITE,

                              &s->interrupt_callback, NULL)) < 0)

            goto fail;

    }



fail:

    if (ret) {

        if (seg->list)

            avio_close(seg->list_pb);

        if (seg->avf)

            avformat_free_context(seg->avf);

    }

    return ret;

}
