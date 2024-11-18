static int pulse_write_packet(AVFormatContext *h, AVPacket *pkt)

{

    PulseData *s = h->priv_data;

    int size     = pkt->size;

    uint8_t *buf = pkt->data;

    int error;



    if (s->stream_index != pkt->stream_index)

        return 0;



    if ((error = pa_simple_write(s->pa, buf, size, &error))) {

        av_log(s, AV_LOG_ERROR, "pa_simple_write failed: %s\n", pa_strerror(error));

        return AVERROR(EIO);

    }



    return 0;

}
