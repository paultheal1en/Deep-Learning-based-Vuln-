static int lrc_read_header(AVFormatContext *s)

{

    LRCContext *lrc = s->priv_data;

    AVBPrint line;

    AVStream *st;



    st = avformat_new_stream(s, NULL);

    if(!st) {

        return AVERROR(ENOMEM);

    }

    avpriv_set_pts_info(st, 64, 1, 1000);

    lrc->ts_offset = 0;

    st->codecpar->codec_type = AVMEDIA_TYPE_SUBTITLE;

    st->codecpar->codec_id   = AV_CODEC_ID_TEXT;

    av_bprint_init(&line, 0, AV_BPRINT_SIZE_UNLIMITED);



    while(!avio_feof(s->pb)) {

        int64_t pos = read_line(&line, s->pb);

        int64_t header_offset = find_header(line.str);

        if(header_offset >= 0) {

            char *comma_offset = strchr(line.str, ':');

            if(comma_offset) {

                char *right_bracket_offset = strchr(line.str, ']');

                if(!right_bracket_offset) {

                    continue;

                }



                *right_bracket_offset = *comma_offset = '\0';

                if(strcmp(line.str + 1, "offset") ||

                   sscanf(comma_offset + 1, "%"SCNd64, &lrc->ts_offset) != 1) {

                    av_dict_set(&s->metadata, line.str + 1, comma_offset + 1, 0);

                }

                *comma_offset = ':';

                *right_bracket_offset = ']';

            }



        } else {

            AVPacket *sub;

            int64_t ts_start = AV_NOPTS_VALUE;

            int64_t ts_stroffset = 0;

            int64_t ts_stroffset_incr = 0;

            int64_t ts_strlength = count_ts(line.str);



            while((ts_stroffset_incr = read_ts(line.str + ts_stroffset,

                                               &ts_start)) != 0) {

                ts_stroffset += ts_stroffset_incr;

                sub = ff_subtitles_queue_insert(&lrc->q, line.str + ts_strlength,

                                                line.len - ts_strlength, 0);

                if(!sub) {

                    return AVERROR(ENOMEM);

                }

                sub->pos = pos;

                sub->pts = ts_start - lrc->ts_offset;

                sub->duration = -1;

            }

        }

    }

    ff_subtitles_queue_finalize(s, &lrc->q);

    ff_metadata_conv_ctx(s, NULL, ff_lrc_metadata_conv);


    return 0;

}