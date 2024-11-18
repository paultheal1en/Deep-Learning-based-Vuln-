static av_cold int mpeg_mc_decode_init(AVCodecContext *avctx){

    if( avctx->thread_count > 1)

        return -1;

    if( !(avctx->slice_flags & SLICE_FLAG_CODED_ORDER) )

        return -1;

    if( !(avctx->slice_flags & SLICE_FLAG_ALLOW_FIELD) ){

        av_dlog(avctx, "mpeg12.c: XvMC decoder will work better if SLICE_FLAG_ALLOW_FIELD is set\n");

    }

    mpeg_decode_init(avctx);



    avctx->pix_fmt = PIX_FMT_XVMC_MPEG2_IDCT;

    avctx->xvmc_acceleration = 2;//2 - the blocks are packed!



    return 0;

}
