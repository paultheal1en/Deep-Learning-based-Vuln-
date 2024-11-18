static int mpegts_handle_packet(AVFormatContext *ctx, PayloadContext *data,

                                AVStream *st, AVPacket *pkt, uint32_t *timestamp,

                                const uint8_t *buf, int len, uint16_t seq,

                                int flags)

{

    int ret;



    // We don't want to use the RTP timestamps at all. If the mpegts demuxer

    // doesn't set any pts/dts, the generic rtpdec code shouldn't try to

    // fill it in either, since the mpegts and RTP timestamps are in totally

    // different ranges.

    *timestamp = RTP_NOTS_VALUE;



    if (!data->ts)

        return AVERROR(EINVAL);



    if (!buf) {

        if (data->read_buf_index >= data->read_buf_size)

            return AVERROR(EAGAIN);

        ret = ff_mpegts_parse_packet(data->ts, pkt, data->buf + data->read_buf_index,

                                     data->read_buf_size - data->read_buf_index);

        if (ret < 0)

            return AVERROR(EAGAIN);

        data->read_buf_index += ret;

        if (data->read_buf_index < data->read_buf_size)

            return 1;

        else

            return 0;

    }



    ret = ff_mpegts_parse_packet(data->ts, pkt, buf, len);

    /* The only error that can be returned from ff_mpegts_parse_packet

     * is "no more data to return from the provided buffer", so return

     * AVERROR(EAGAIN) for all errors */

    if (ret < 0)

        return AVERROR(EAGAIN);

    if (ret < len) {

        data->read_buf_size = FFMIN(len - ret, sizeof(data->buf));

        memcpy(data->buf, buf + ret, data->read_buf_size);

        data->read_buf_index = 0;

        return 1;

    }

    return 0;

}
