static void apply_param_change(AVCodecContext *avctx, AVPacket *avpkt)

{

    int size = 0;

    const uint8_t *data;

    uint32_t flags;



    if (!(avctx->codec->capabilities & CODEC_CAP_PARAM_CHANGE))

        return;



    data = av_packet_get_side_data(avpkt, AV_PKT_DATA_PARAM_CHANGE, &size);

    if (!data || size < 4)

        return;

    flags = bytestream_get_le32(&data);

    size -= 4;

    if (size < 4) /* Required for any of the changes */

        return;

    if (flags & AV_SIDE_DATA_PARAM_CHANGE_CHANNEL_COUNT) {

        avctx->channels = bytestream_get_le32(&data);

        size -= 4;

    }

    if (flags & AV_SIDE_DATA_PARAM_CHANGE_CHANNEL_LAYOUT) {

        if (size < 8)

            return;

        avctx->channel_layout = bytestream_get_le64(&data);

        size -= 8;

    }

    if (size < 4)

        return;

    if (flags & AV_SIDE_DATA_PARAM_CHANGE_SAMPLE_RATE) {

        avctx->sample_rate = bytestream_get_le32(&data);

        size -= 4;

    }

    if (flags & AV_SIDE_DATA_PARAM_CHANGE_DIMENSIONS) {

        if (size < 8)

            return;

        avctx->width  = bytestream_get_le32(&data);

        avctx->height = bytestream_get_le32(&data);

        avcodec_set_dimensions(avctx, avctx->width, avctx->height);

        size -= 8;

    }

}
