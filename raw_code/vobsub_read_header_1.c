static int vobsub_read_header(AVFormatContext *s)

{

    int i, ret = 0, header_parsed = 0, langidx = 0;

    MpegDemuxContext *vobsub = s->priv_data;

    char *sub_name = NULL;

    size_t fname_len;

    char *ext, *header_str;

    AVBPrint header;

    int64_t delay = 0;

    AVStream *st = NULL;



    sub_name = av_strdup(s->filename);

    fname_len = strlen(sub_name);

    ext = sub_name - 3 + fname_len;

    if (fname_len < 4 || *(ext - 1) != '.') {

        av_log(s, AV_LOG_ERROR, "The input index filename is too short "

               "to guess the associated .SUB file\n");

        ret = AVERROR_INVALIDDATA;

        goto end;

    }

    memcpy(ext, !strncmp(ext, "IDX", 3) ? "SUB" : "sub", 3);

    av_log(s, AV_LOG_VERBOSE, "IDX/SUB: %s -> %s\n", s->filename, sub_name);

    ret = avformat_open_input(&vobsub->sub_ctx, sub_name, &ff_mpegps_demuxer, NULL);

    if (ret < 0) {

        av_log(s, AV_LOG_ERROR, "Unable to open %s as MPEG subtitles\n", sub_name);

        goto end;

    }



    av_bprint_init(&header, 0, AV_BPRINT_SIZE_UNLIMITED);

    while (!url_feof(s->pb)) {

        char line[2048];

        int len = ff_get_line(s->pb, line, sizeof(line));



        if (!len)

            break;



        line[strcspn(line, "\r\n")] = 0;



        if (!strncmp(line, "id:", 3)) {

            int n, stream_id = 0;

            char id[64] = {0};



            n = sscanf(line, "id: %63[^,], index: %u", id, &stream_id);

            if (n != 2) {

                av_log(s, AV_LOG_WARNING, "Unable to parse index line '%s', "

                       "assuming 'id: und, index: 0'\n", line);

                strcpy(id, "und");

                stream_id = 0;

            }



            if (stream_id >= FF_ARRAY_ELEMS(vobsub->q)) {

                av_log(s, AV_LOG_ERROR, "Maximum number of subtitles streams reached\n");

                ret = AVERROR(EINVAL);

                goto end;

            }



            st = avformat_new_stream(s, NULL);

            if (!st) {

                ret = AVERROR(ENOMEM);

                goto end;

            }

            st->id = stream_id;

            st->codec->codec_type = AVMEDIA_TYPE_SUBTITLE;

            st->codec->codec_id   = AV_CODEC_ID_DVD_SUBTITLE;

            avpriv_set_pts_info(st, 64, 1, 1000);

            av_dict_set(&st->metadata, "language", id, 0);

            av_log(s, AV_LOG_DEBUG, "IDX stream[%d] id=%s\n", stream_id, id);

            header_parsed = 1;



        } else if (st && !strncmp(line, "timestamp:", 10)) {

            AVPacket *sub;

            int hh, mm, ss, ms;

            int64_t pos, timestamp;

            const char *p = line + 10;



            if (!s->nb_streams) {

                av_log(s, AV_LOG_ERROR, "Timestamp declared before any stream\n");

                ret = AVERROR_INVALIDDATA;

                goto end;

            }



            if (sscanf(p, "%02d:%02d:%02d:%03d, filepos: %"SCNx64,

                       &hh, &mm, &ss, &ms, &pos) != 5) {

                av_log(s, AV_LOG_ERROR, "Unable to parse timestamp line '%s', "

                       "abort parsing\n", line);

                break;

            }

            timestamp = (hh*3600LL + mm*60LL + ss) * 1000LL + ms + delay;

            timestamp = av_rescale_q(timestamp, av_make_q(1, 1000), st->time_base);



            sub = ff_subtitles_queue_insert(&vobsub->q[s->nb_streams - 1], "", 0, 0);

            if (!sub) {

                ret = AVERROR(ENOMEM);

                goto end;

            }

            sub->pos = pos;

            sub->pts = timestamp;

            sub->stream_index = s->nb_streams - 1;



        } else if (st && !strncmp(line, "alt:", 4)) {

            const char *p = line + 4;



            while (*p == ' ')

                p++;

            av_dict_set(&st->metadata, "title", p, 0);

            av_log(s, AV_LOG_DEBUG, "IDX stream[%d] name=%s\n", st->id, p);

            header_parsed = 1;



        } else if (!strncmp(line, "delay:", 6)) {

            int sign = 1, hh = 0, mm = 0, ss = 0, ms = 0;

            const char *p = line + 6;



            while (*p == ' ')

                p++;

            if (*p == '-' || *p == '+') {

                sign = *p == '-' ? -1 : 1;

                p++;

            }

            sscanf(p, "%d:%d:%d:%d", &hh, &mm, &ss, &ms);

            delay = ((hh*3600LL + mm*60LL + ss) * 1000LL + ms) * sign;



        } else if (!strncmp(line, "langidx:", 8)) {

            const char *p = line + 8;



            if (sscanf(p, "%d", &langidx) != 1)

                av_log(s, AV_LOG_ERROR, "Invalid langidx specified\n");



        } else if (!header_parsed) {

            if (line[0] && line[0] != '#')

                av_bprintf(&header, "%s\n", line);

        }

    }



    if (langidx < s->nb_streams)

        s->streams[langidx]->disposition |= AV_DISPOSITION_DEFAULT;



    for (i = 0; i < s->nb_streams; i++) {

        vobsub->q[i].sort = SUB_SORT_POS_TS;

        ff_subtitles_queue_finalize(&vobsub->q[i]);

    }



    if (!av_bprint_is_complete(&header)) {

        av_bprint_finalize(&header, NULL);

        ret = AVERROR(ENOMEM);

        goto end;

    }

    av_bprint_finalize(&header, &header_str);

    for (i = 0; i < s->nb_streams; i++) {

        AVStream *sub_st = s->streams[i];

        sub_st->codec->extradata      = av_strdup(header_str);

        sub_st->codec->extradata_size = header.len;

    }

    av_free(header_str);



end:

    av_free(sub_name);

    return ret;

}
