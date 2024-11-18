static AVStream *get_subtitle_pkt(AVFormatContext *s, AVStream *next_st,

                                  AVPacket *pkt)

{

    AVIStream *ast, *next_ast = next_st->priv_data;

    int64_t ts, next_ts, ts_min = INT64_MAX;

    AVStream *st, *sub_st = NULL;

    int i;



    next_ts = av_rescale_q(next_ast->frame_offset, next_st->time_base,

                           AV_TIME_BASE_Q);



    for (i=0; i<s->nb_streams; i++) {

        st  = s->streams[i];

        ast = st->priv_data;

        if (st->discard < AVDISCARD_ALL && ast->sub_pkt.data) {

            ts = av_rescale_q(ast->sub_pkt.dts, st->time_base, AV_TIME_BASE_Q);

            if (ts <= next_ts && ts < ts_min) {

                ts_min = ts;

                sub_st = st;

            }

        }

    }



    if (sub_st) {

        ast = sub_st->priv_data;

        *pkt = ast->sub_pkt;

        pkt->stream_index = sub_st->index;

        if (av_read_packet(ast->sub_ctx, &ast->sub_pkt) < 0)

            ast->sub_pkt.data = NULL;

    }

    return sub_st;

}
