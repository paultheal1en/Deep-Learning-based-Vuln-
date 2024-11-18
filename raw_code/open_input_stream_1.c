static int open_input_stream(HTTPContext *c, const char *info)

{

    char buf[128];

    char input_filename[1024];

    AVFormatContext *s = NULL;

    int buf_size, i, ret;

    int64_t stream_pos;



    /* find file name */

    if (c->stream->feed) {

        strcpy(input_filename, c->stream->feed->feed_filename);

        buf_size = FFM_PACKET_SIZE;

        /* compute position (absolute time) */

        if (av_find_info_tag(buf, sizeof(buf), "date", info)) {

            if ((ret = av_parse_time(&stream_pos, buf, 0)) < 0) {

                http_log("Invalid date specification '%s' for stream\n", buf);

                return ret;

            }

        } else if (av_find_info_tag(buf, sizeof(buf), "buffer", info)) {

            int prebuffer = strtol(buf, 0, 10);

            stream_pos = av_gettime() - prebuffer * (int64_t)1000000;

        } else

            stream_pos = av_gettime() - c->stream->prebuffer * (int64_t)1000;

    } else {

        strcpy(input_filename, c->stream->feed_filename);

        buf_size = 0;

        /* compute position (relative time) */

        if (av_find_info_tag(buf, sizeof(buf), "date", info)) {

            if ((ret = av_parse_time(&stream_pos, buf, 1)) < 0) {

                http_log("Invalid date specification '%s' for stream\n", buf);

                return ret;

            }

        } else

            stream_pos = 0;

    }

    if (!input_filename[0]) {

        http_log("No filename was specified for stream\n");

        return AVERROR(EINVAL);

    }



    /* open stream */

    ret = avformat_open_input(&s, input_filename, c->stream->ifmt,

                              &c->stream->in_opts);

    if (ret < 0) {

        http_log("Could not open input '%s': %s\n",

                 input_filename, av_err2str(ret));

        return ret;

    }



    /* set buffer size */

    if (buf_size > 0) {

        ret = ffio_set_buf_size(s->pb, buf_size);

        if (ret < 0) {

            http_log("Failed to set buffer size\n");

            return ret;

        }

    }



    s->flags |= AVFMT_FLAG_GENPTS;

    c->fmt_in = s;

    if (strcmp(s->iformat->name, "ffm") &&

        (ret = avformat_find_stream_info(c->fmt_in, NULL)) < 0) {

        http_log("Could not find stream info for input '%s'\n", input_filename);

        avformat_close_input(&s);

        return ret;

    }



    /* choose stream as clock source (we favor the video stream if

     * present) for packet sending */

    c->pts_stream_index = 0;

    for(i=0;i<c->stream->nb_streams;i++) {

        if (c->pts_stream_index == 0 &&

            c->stream->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {

            c->pts_stream_index = i;

        }

    }



    if (c->fmt_in->iformat->read_seek)

        av_seek_frame(c->fmt_in, -1, stream_pos, 0);

    /* set the start time (needed for maxtime and RTP packet timing) */

    c->start_time = cur_time;

    c->first_pts = AV_NOPTS_VALUE;

    return 0;

}
