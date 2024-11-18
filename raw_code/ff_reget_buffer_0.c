int ff_reget_buffer(AVCodecContext *avctx, AVFrame *frame)

{

    AVFrame *tmp;

    int ret;



    av_assert0(avctx->codec_type == AVMEDIA_TYPE_VIDEO);



    if (!frame->data[0])

        return ff_get_buffer(avctx, frame, AV_GET_BUFFER_FLAG_REF);



    if (av_frame_is_writable(frame)) {

        frame->pkt_pts = avctx->internal->pkt ? avctx->internal->pkt->pts : AV_NOPTS_VALUE;

        frame->reordered_opaque = avctx->reordered_opaque;

        return 0;

    }



    tmp = av_frame_alloc();

    if (!tmp)

        return AVERROR(ENOMEM);



    av_frame_move_ref(tmp, frame);



    ret = ff_get_buffer(avctx, frame, AV_GET_BUFFER_FLAG_REF);

    if (ret < 0) {

        av_frame_free(&tmp);

        return ret;

    }



    av_frame_copy(frame, tmp);

    av_frame_free(&tmp);



    return 0;

}
