static int apply_param_change(AVCodecContext *avctx, AVPacket *avpkt)

{

    int size = 0, ret;

    const uint8_t *data;

    uint32_t flags;



    data = av_packet_get_side_data(avpkt, AV_PKT_DATA_PARAM_CHANGE, &size);

    if (!data)

        return 0;



    if (!(avctx->codec->capabilities & CODEC_CAP_PARAM_CHANGE)) {

        av_log(avctx, AV_LOG_ERROR, "This decoder does not support parameter "

               "changes, but PARAM_CHANGE side data was sent to it.\n");

        return AVERROR(EINVAL);

    }



    if (size < 4)

        goto fail;



    flags = bytestream_get_le32(&data);

    size -= 4;



    if (flags & AV_SIDE_DATA_PARAM_CHANGE_CHANNEL_COUNT) {

        if (size < 4)

            goto fail;

        avctx->channels = bytestream_get_le32(&data);

        size -= 4;

    }

    if (flags & AV_SIDE_DATA_PARAM_CHANGE_CHANNEL_LAYOUT) {

        if (size < 8)

            goto fail;

        avctx->channel_layout = bytestream_get_le64(&data);

        size -= 8;

    }

    if (flags & AV_SIDE_DATA_PARAM_CHANGE_SAMPLE_RATE) {

        if (size < 4)

            goto fail;

        avctx->sample_rate = bytestream_get_le32(&data);

        size -= 4;

    }

    if (flags & AV_SIDE_DATA_PARAM_CHANGE_DIMENSIONS) {

        if (size < 8)

            goto fail;

        avctx->width  = bytestream_get_le32(&data);

        avctx->height = bytestream_get_le32(&data);

        size -= 8;

        ret = ff_set_dimensions(avctx, avctx->width, avctx->height);

        if (ret < 0)

            return ret;

    }



    return 0;

fail:

    av_log(avctx, AV_LOG_ERROR, "PARAM_CHANGE side data too small.\n");

    return AVERROR_INVALIDDATA;

}
