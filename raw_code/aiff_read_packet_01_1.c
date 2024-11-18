static int aiff_read_packet(AVFormatContext *s,
                            AVPacket *pkt)
{
    AVStream *st = s->streams[0];
    AIFFInputContext *aiff = s->priv_data;
    int64_t max_size;
    int res, size;
    /* calculate size of remaining data */
    max_size = aiff->data_end - avio_tell(s->pb);
    if (max_size <= 0)
        return AVERROR_EOF;
    /* Now for that packet */
    switch (st->codecpar->codec_id) {
    case AV_CODEC_ID_ADPCM_IMA_QT:
    case AV_CODEC_ID_GSM:
    case AV_CODEC_ID_QDM2:
    case AV_CODEC_ID_QCELP:
        size = st->codecpar->block_align;
        break;
    default:
        size = st->codecpar->block_align ? (MAX_SIZE / st->codecpar->block_align) * st->codecpar->block_align : MAX_SIZE;
    size = FFMIN(max_size, size);
    res = av_get_packet(s->pb, pkt, size);
    if (res < 0)
        return res;
    if (size >= st->codecpar->block_align)
        pkt->flags &= ~AV_PKT_FLAG_CORRUPT;
    /* Only one stream in an AIFF file */
    pkt->stream_index = 0;
    pkt->duration     = (res / st->codecpar->block_align) * aiff->block_duration;
    return 0;