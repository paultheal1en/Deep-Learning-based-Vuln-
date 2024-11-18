static int wrapped_avframe_encode(AVCodecContext *avctx, AVPacket *pkt,

                     const AVFrame *frame, int *got_packet)

{

    AVFrame *wrapped = av_frame_clone(frame);



    if (!wrapped)

        return AVERROR(ENOMEM);



    pkt->buf = av_buffer_create((uint8_t *)wrapped, sizeof(*wrapped),

                                wrapped_avframe_release_buffer, NULL,

                                AV_BUFFER_FLAG_READONLY);

    if (!pkt->buf) {

        av_frame_free(&wrapped);

        return AVERROR(ENOMEM);

    }



    pkt->data = (uint8_t *)wrapped;

    pkt->size = sizeof(*wrapped);



    pkt->flags |= AV_PKT_FLAG_KEY;

    *got_packet = 1;

    return 0;

}
