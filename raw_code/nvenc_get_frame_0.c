static int nvenc_get_frame(AVCodecContext *avctx, AVPacket *pkt)

{

    NVENCContext *ctx               = avctx->priv_data;

    NV_ENCODE_API_FUNCTION_LIST *nv = &ctx->nvel.nvenc_funcs;

    NV_ENC_LOCK_BITSTREAM params    = { 0 };

    NVENCOutputSurface *out         = NULL;

    int ret;



    ret = nvenc_dequeue_surface(ctx->pending, &out);

    if (ret)

        return ret;



    params.version         = NV_ENC_LOCK_BITSTREAM_VER;

    params.outputBitstream = out->out;



    ret = nv->nvEncLockBitstream(ctx->nvenc_ctx, &params);

    if (ret < 0)

        return nvenc_print_error(avctx, ret, "Cannot lock the bitstream");



    ret = ff_alloc_packet(pkt, params.bitstreamSizeInBytes);

    if (ret < 0)

        return ret;



    memcpy(pkt->data, params.bitstreamBufferPtr, pkt->size);



    ret = nv->nvEncUnlockBitstream(ctx->nvenc_ctx, out->out);

    if (ret < 0)

        return nvenc_print_error(avctx, ret, "Cannot unlock the bitstream");



    out->busy = out->in->locked = 0;



    ret = nvenc_set_timestamp(ctx, &params, pkt);

    if (ret < 0)

        return ret;



    switch (params.pictureType) {

    case NV_ENC_PIC_TYPE_IDR:

        pkt->flags |= AV_PKT_FLAG_KEY;

#if FF_API_CODED_FRAME

FF_DISABLE_DEPRECATION_WARNINGS

    case NV_ENC_PIC_TYPE_INTRA_REFRESH:

    case NV_ENC_PIC_TYPE_I:

        avctx->coded_frame->pict_type = AV_PICTURE_TYPE_I;

        break;

    case NV_ENC_PIC_TYPE_P:

        avctx->coded_frame->pict_type = AV_PICTURE_TYPE_P;

        break;

    case NV_ENC_PIC_TYPE_B:

        avctx->coded_frame->pict_type = AV_PICTURE_TYPE_B;

        break;

    case NV_ENC_PIC_TYPE_BI:

        avctx->coded_frame->pict_type = AV_PICTURE_TYPE_BI;

        break;

FF_ENABLE_DEPRECATION_WARNINGS

#endif

    }



    return 0;

}
