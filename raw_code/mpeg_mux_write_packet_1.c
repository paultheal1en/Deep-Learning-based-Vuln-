static int mpeg_mux_write_packet(AVFormatContext *ctx, AVPacket *pkt)

{

    int stream_index = pkt->stream_index;

    int size         = pkt->size;

    uint8_t *buf     = pkt->data;

    MpegMuxContext *s = ctx->priv_data;

    AVStream *st      = ctx->streams[stream_index];

    StreamInfo *stream = st->priv_data;

    int64_t pts, dts;

    PacketDesc *pkt_desc;

    int preload;

    const int is_iframe = st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO &&

                          (pkt->flags & AV_PKT_FLAG_KEY);



    preload = av_rescale(s->preload, 90000, AV_TIME_BASE);



    pts = pkt->pts;

    dts = pkt->dts;



    if (s->last_scr == AV_NOPTS_VALUE) {

        if (dts == AV_NOPTS_VALUE || (dts < preload && ctx->avoid_negative_ts) || s->is_dvd) {

            if (dts != AV_NOPTS_VALUE)

                s->preload += av_rescale(-dts, AV_TIME_BASE, 90000);

            s->last_scr = 0;

        } else {

            s->last_scr = dts - preload;

            s->preload = 0;

        }

        preload = av_rescale(s->preload, 90000, AV_TIME_BASE);

        av_log(ctx, AV_LOG_DEBUG, "First SCR: %"PRId64" First DTS: %"PRId64"\n", s->last_scr, dts + preload);

    }



    if (dts != AV_NOPTS_VALUE) dts += preload;

    if (pts != AV_NOPTS_VALUE) pts += preload;



    av_log(ctx, AV_LOG_TRACE, "dts:%f pts:%f flags:%d stream:%d nopts:%d\n",

            dts / 90000.0, pts / 90000.0, pkt->flags,

            pkt->stream_index, pts != AV_NOPTS_VALUE);

    if (!stream->premux_packet)

        stream->next_packet = &stream->premux_packet;

    *stream->next_packet     =

    pkt_desc                 = av_mallocz(sizeof(PacketDesc));



    pkt_desc->pts            = pts;

    pkt_desc->dts            = dts;

    pkt_desc->unwritten_size =

    pkt_desc->size           = size;

    if (!stream->predecode_packet)

        stream->predecode_packet = pkt_desc;

    stream->next_packet = &pkt_desc->next;



    if (av_fifo_realloc2(stream->fifo, av_fifo_size(stream->fifo) + size) < 0)

        return -1;



    if (s->is_dvd) {

        // min VOBU length 0.4 seconds (mpucoder)

        if (is_iframe &&

            (s->packet_number == 0 ||

             (pts - stream->vobu_start_pts >= 36000))) {

            stream->bytes_to_iframe = av_fifo_size(stream->fifo);

            stream->align_iframe    = 1;

            stream->vobu_start_pts  = pts;

        }

    }



    av_fifo_generic_write(stream->fifo, buf, size, NULL);



    for (;;) {

        int ret = output_packet(ctx, 0);

        if (ret <= 0)

            return ret;

    }

}