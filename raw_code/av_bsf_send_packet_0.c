int av_bsf_send_packet(AVBSFContext *ctx, AVPacket *pkt)

{

    if (!pkt) {

        ctx->internal->eof = 1;

        return 0;

    }



    if (ctx->internal->eof) {

        av_log(ctx, AV_LOG_ERROR, "A non-NULL packet sent after an EOF.\n");

        return AVERROR(EINVAL);

    }



    if (ctx->internal->buffer_pkt->data ||

        ctx->internal->buffer_pkt->side_data_elems)

        return AVERROR(EAGAIN);



    av_packet_move_ref(ctx->internal->buffer_pkt, pkt);



    return 0;

}
