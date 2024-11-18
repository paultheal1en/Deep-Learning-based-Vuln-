static int svc_decode_frame(AVCodecContext *avctx, void *data,

                            int *got_frame, AVPacket *avpkt)

{

    SVCContext *s = avctx->priv_data;

    SBufferInfo info = { 0 };

    uint8_t* ptrs[3];

    int linesize[3];

    AVFrame *avframe = data;

    DECODING_STATE state;



    state = (*s->decoder)->DecodeFrame2(s->decoder, avpkt->data, avpkt->size, ptrs, &info);

    if (state != dsErrorFree) {

        av_log(avctx, AV_LOG_ERROR, "DecodeFrame2 failed\n");

        return AVERROR_UNKNOWN;

    }

    if (info.iBufferStatus != 1) {

        av_log(avctx, AV_LOG_DEBUG, "No frame produced\n");

        return avpkt->size;

    }



    ff_set_dimensions(avctx, info.UsrData.sSystemBuffer.iWidth, info.UsrData.sSystemBuffer.iHeight);

    // The decoder doesn't (currently) support decoding into a user

    // provided buffer, so do a copy instead.

    if (ff_get_buffer(avctx, avframe, 0) < 0) {

        av_log(avctx, AV_LOG_ERROR, "Unable to allocate buffer\n");

        return AVERROR(ENOMEM);

    }



    linesize[0] = info.UsrData.sSystemBuffer.iStride[0];

    linesize[1] = linesize[2] = info.UsrData.sSystemBuffer.iStride[1];

    av_image_copy(avframe->data, avframe->linesize, (const uint8_t **) ptrs, linesize, avctx->pix_fmt, avctx->width, avctx->height);



    avframe->pts     = avpkt->pts;

    avframe->pkt_dts = avpkt->dts;

#if FF_API_PKT_PTS

FF_DISABLE_DEPRECATION_WARNINGS

    avframe->pkt_pts = avpkt->pts;

FF_ENABLE_DEPRECATION_WARNINGS

#endif



    *got_frame = 1;

    return avpkt->size;

}
