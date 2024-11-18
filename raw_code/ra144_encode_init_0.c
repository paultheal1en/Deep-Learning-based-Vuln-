static av_cold int ra144_encode_init(AVCodecContext * avctx)

{

    RA144Context *ractx;

    int ret;



    if (avctx->channels != 1) {

        av_log(avctx, AV_LOG_ERROR, "invalid number of channels: %d\n",

               avctx->channels);

        return -1;

    }

    avctx->frame_size = NBLOCKS * BLOCKSIZE;

    avctx->delay      = avctx->frame_size;

    avctx->bit_rate = 8000;

    ractx = avctx->priv_data;

    ractx->lpc_coef[0] = ractx->lpc_tables[0];

    ractx->lpc_coef[1] = ractx->lpc_tables[1];

    ractx->avctx = avctx;

    ret = ff_lpc_init(&ractx->lpc_ctx, avctx->frame_size, LPC_ORDER,

                      FF_LPC_TYPE_LEVINSON);

    if (ret < 0)

        goto error;



    ff_af_queue_init(avctx, &ractx->afq);



    return 0;

error:

    ra144_encode_close(avctx);

    return ret;

}
