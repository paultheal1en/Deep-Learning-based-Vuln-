static int smjpeg_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    SMJPEGContext *sc = s->priv_data;

    uint32_t dtype, ret, size, timestamp;

    int64_t pos;



    if (s->pb->eof_reached)

        return AVERROR_EOF;

    pos   = avio_tell(s->pb);

    dtype = avio_rl32(s->pb);

    switch (dtype) {

    case SMJPEG_SNDD:

        timestamp = avio_rb32(s->pb);

        size = avio_rb32(s->pb);

        ret = av_get_packet(s->pb, pkt, size);

        pkt->stream_index = sc->audio_stream_index;

        pkt->pts = timestamp;

        pkt->pos = pos;

        break;

    case SMJPEG_VIDD:

        timestamp = avio_rb32(s->pb);

        size = avio_rb32(s->pb);

        ret = av_get_packet(s->pb, pkt, size);

        pkt->stream_index = sc->video_stream_index;

        pkt->pts = timestamp;

        pkt->pos = pos;

        break;

    case SMJPEG_DONE:

        ret = AVERROR_EOF;

        break;

    default:

        av_log(s, AV_LOG_ERROR, "unknown chunk %x\n", dtype);

        ret = AVERROR_INVALIDDATA;

        break;

    }

    return ret;

}
