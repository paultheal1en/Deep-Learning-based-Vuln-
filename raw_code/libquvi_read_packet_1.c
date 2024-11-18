static int libquvi_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    LibQuviContext *qc = s->priv_data;

    return av_read_frame(qc->fmtctx, pkt);

}
