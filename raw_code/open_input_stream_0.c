static int open_input_stream(HTTPContext *c, const char *info)

{

    char buf[128];

    char input_filename[1024];

    AVFormatContext *s;

    int buf_size, i, ret;

    int64_t stream_pos;



    /* find file name */

    if (c->stream->feed) {

        strcpy(input_filename, c->stream->feed->feed_filename);

        buf_size = FFM_PACKET_SIZE;

        /* compute position (absolute time) */

        if (find_info_tag(buf, sizeof(buf), "date", info)) {

            stream_pos = parse_date(buf, 0);

            if (stream_pos == INT64_MIN)

                return -1;

        } else if (find_info_tag(buf, sizeof(buf), "buffer", info)) {

            int prebuffer = strtol(buf, 0, 10);

            stream_pos = av_gettime() - prebuffer * (int64_t)1000000;

        } else

            stream_pos = av_gettime() - c->stream->prebuffer * (int64_t)1000;

    } else {

        strcpy(input_filename, c->stream->feed_filename);

        buf_size = 0;

        /* compute position (relative time) */

        if (find_info_tag(buf, sizeof(buf), "date", info)) {

            stream_pos = parse_date(buf, 1);

            if (stream_pos == INT64_MIN)

                return -1;

        } else

            stream_pos = 0;

    }

    if (input_filename[0] == '\0')

        return -1;



#if 0

    { time_t when = stream_pos / 1000000;

    http_log("Stream pos = %"PRId64", time=%s", stream_pos, ctime(&when));

    }

#endif



    /* open stream */

    if ((ret = av_open_input_file(&s, input_filename, c->stream->ifmt,

                                  buf_size, c->stream->ap_in)) < 0) {

        http_log("could not open %s: %d\n", input_filename, ret);

        return -1;

    }

    s->flags |= AVFMT_FLAG_GENPTS;

    c->fmt_in = s;

    av_find_stream_info(c->fmt_in);



    /* open each parser */

    for(i=0;i<s->nb_streams;i++)

        open_parser(s, i);



    /* choose stream as clock source (we favorize video stream if

       present) for packet sending */

    c->pts_stream_index = 0;

    for(i=0;i<c->stream->nb_streams;i++) {

        if (c->pts_stream_index == 0 &&

            c->stream->streams[i]->codec->codec_type == CODEC_TYPE_VIDEO) {

            c->pts_stream_index = i;

        }

    }



#if 1

    if (c->fmt_in->iformat->read_seek)

        av_seek_frame(c->fmt_in, -1, stream_pos, 0);

#endif

    /* set the start time (needed for maxtime and RTP packet timing) */

    c->start_time = cur_time;

    c->first_pts = AV_NOPTS_VALUE;

    return 0;

}
