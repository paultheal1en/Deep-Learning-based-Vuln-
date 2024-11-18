static int adx_read_packet(AVFormatContext *s, AVPacket *pkt)
{
    ADXDemuxerContext *c = s->priv_data;
    AVCodecContext *avctx = s->streams[0]->codec;
    int ret, size;
    size = BLOCK_SIZE * avctx->channels;
    pkt->pos = avio_tell(s->pb);
    pkt->stream_index = 0;
    ret = av_get_packet(s->pb, pkt, size);
    if (ret != size) {
        av_free_packet(pkt);
        return ret < 0 ? ret : AVERROR(EIO);
    if (AV_RB16(pkt->data) & 0x8000) {
        av_free_packet(pkt);
        return AVERROR_EOF;
    pkt->size     = size;
    pkt->duration = 1;
    pkt->pts      = (pkt->pos - c->header_size) / size;
    return 0;