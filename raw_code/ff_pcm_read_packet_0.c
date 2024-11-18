int ff_pcm_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    int ret, size;



    size= RAW_SAMPLES*s->streams[0]->codec->block_align;

    if (size <= 0)

        return AVERROR(EINVAL);



    ret= av_get_packet(s->pb, pkt, size);



    pkt->flags &= ~AV_PKT_FLAG_CORRUPT;

    pkt->stream_index = 0;

    if (ret < 0)

        return ret;



    return ret;

}
