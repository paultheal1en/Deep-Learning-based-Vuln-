static void show_packet(WriterContext *w, AVFormatContext *fmt_ctx, AVPacket *pkt, int packet_idx)

{

    char val_str[128];

    AVStream *st = fmt_ctx->streams[pkt->stream_index];

    struct print_buf pbuf = {.s = NULL};



    print_section_header("packet");

    print_str("codec_type",       av_x_if_null(av_get_media_type_string(st->codec->codec_type), "unknown"));

    print_int("stream_index",     pkt->stream_index);

    print_ts  ("pts",             pkt->pts);

    print_time("pts_time",        pkt->pts, &st->time_base);

    print_ts  ("dts",             pkt->dts);

    print_time("dts_time",        pkt->dts, &st->time_base);

    print_ts  ("duration",        pkt->duration);

    print_time("duration_time",   pkt->duration, &st->time_base);

    print_val("size",             pkt->size, unit_byte_str);

    print_fmt("pos",   "%"PRId64, pkt->pos);

    print_fmt("flags", "%c",      pkt->flags & AV_PKT_FLAG_KEY ? 'K' : '_');

    print_section_footer("packet");



    av_free(pbuf.s);

    fflush(stdout);

}
