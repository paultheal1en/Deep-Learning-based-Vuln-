int ff_write_chained(AVFormatContext *dst, int dst_stream, AVPacket *pkt,

                     AVFormatContext *src)

{

    AVPacket local_pkt;



    local_pkt = *pkt;

    local_pkt.stream_index = dst_stream;

    if (pkt->pts != AV_NOPTS_VALUE)

        local_pkt.pts = av_rescale_q(pkt->pts,

                                     src->streams[pkt->stream_index]->time_base,

                                     dst->streams[dst_stream]->time_base);

    if (pkt->dts != AV_NOPTS_VALUE)

        local_pkt.dts = av_rescale_q(pkt->dts,

                                     src->streams[pkt->stream_index]->time_base,

                                     dst->streams[dst_stream]->time_base);

    if (pkt->duration)

        local_pkt.duration = av_rescale_q(pkt->duration,

                                          src->streams[pkt->stream_index]->time_base,

                                          dst->streams[dst_stream]->time_base);

    return av_write_frame(dst, &local_pkt);

}
