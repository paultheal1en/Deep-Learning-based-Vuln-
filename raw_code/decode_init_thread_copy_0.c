static int decode_init_thread_copy(AVCodecContext *avctx)

{

    H264Context *h = avctx->priv_data;

    int ret;



    if (!avctx->internal->is_copy)

        return 0;

    memset(h->sps_buffers, 0, sizeof(h->sps_buffers));

    memset(h->pps_buffers, 0, sizeof(h->pps_buffers));



    ret = h264_init_context(avctx, h);

    if (ret < 0)

        return ret;



    h->context_initialized = 0;



    return 0;

}
