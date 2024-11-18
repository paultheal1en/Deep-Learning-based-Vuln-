static int v4l2_receive_packet(AVCodecContext *avctx, AVPacket *avpkt)

{

    V4L2m2mContext *s = avctx->priv_data;

    V4L2Context *const capture = &s->capture;

    V4L2Context *const output = &s->output;

    int ret;



    if (s->draining)

        goto dequeue;



    if (!output->streamon) {

        ret = ff_v4l2_context_set_status(output, VIDIOC_STREAMON);

        if (ret) {

            av_log(avctx, AV_LOG_ERROR, "VIDIOC_STREAMOFF failed on output context\n");

            return ret;

        }

    }



    if (!capture->streamon) {

        ret = ff_v4l2_context_set_status(capture, VIDIOC_STREAMON);

        if (ret) {

            av_log(avctx, AV_LOG_ERROR, "VIDIOC_STREAMON failed on capture context\n");

            return ret;

        }

    }



dequeue:

    return ff_v4l2_context_dequeue_packet(capture, avpkt);

}
