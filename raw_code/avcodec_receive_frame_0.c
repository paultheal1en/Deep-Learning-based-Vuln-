int attribute_align_arg avcodec_receive_frame(AVCodecContext *avctx, AVFrame *frame)

{

    int ret;



    av_frame_unref(frame);



    if (!avcodec_is_open(avctx) || !av_codec_is_decoder(avctx->codec))

        return AVERROR(EINVAL);



    if (avctx->codec->receive_frame) {

        if (avctx->internal->draining && !(avctx->codec->capabilities & AV_CODEC_CAP_DELAY))

            return AVERROR_EOF;

        return avctx->codec->receive_frame(avctx, frame);

    }



    // Emulation via old API.



    if (!avctx->internal->buffer_frame->buf[0]) {

        if (!avctx->internal->buffer_pkt->size && !avctx->internal->draining)

            return AVERROR(EAGAIN);



        while (1) {

            if ((ret = do_decode(avctx, avctx->internal->buffer_pkt)) < 0) {

                av_packet_unref(avctx->internal->buffer_pkt);

                return ret;

            }

            // Some audio decoders may consume partial data without returning

            // a frame (fate-wmapro-2ch). There is no way to make the caller

            // call avcodec_receive_frame() again without returning a frame,

            // so try to decode more in these cases.

            if (avctx->internal->buffer_frame->buf[0] ||

                !avctx->internal->buffer_pkt->size)

                break;

        }

    }



    if (!avctx->internal->buffer_frame->buf[0])

        return avctx->internal->draining ? AVERROR_EOF : AVERROR(EAGAIN);



    av_frame_move_ref(frame, avctx->internal->buffer_frame);

    return 0;

}
