static av_cold int prores_encode_init(AVCodecContext *avctx)

{

    int i;

    ProresContext* ctx = avctx->priv_data;



    if (avctx->pix_fmt != PIX_FMT_YUV422P10LE) {

        av_log(avctx, AV_LOG_ERROR, "need YUV422P10\n");

        return -1;

    }

    if (avctx->width & 0x1) {

        av_log(avctx, AV_LOG_ERROR,

                "frame width needs to be multiple of 2\n");

        return -1;

    }



    if ((avctx->height & 0xf) || (avctx->width & 0xf)) {

        ctx->fill_y = av_malloc(DEFAULT_SLICE_MB_WIDTH << 9);

        ctx->fill_u = av_malloc(DEFAULT_SLICE_MB_WIDTH << 8);

        ctx->fill_v = av_malloc(DEFAULT_SLICE_MB_WIDTH << 8);

    }



    if (avctx->profile == FF_PROFILE_UNKNOWN) {

        avctx->profile = FF_PROFILE_PRORES_STANDARD;

        av_log(avctx, AV_LOG_INFO,

                "encoding with ProRes standard (apcn) profile\n");



    } else if (avctx->profile < FF_PROFILE_PRORES_PROXY

            || avctx->profile > FF_PROFILE_PRORES_HQ) {

        av_log(

                avctx,

                AV_LOG_ERROR,

                "unknown profile %d, use [0 - apco, 1 - apcs, 2 - apcn (default), 3 - apch]\n",

                avctx->profile);

        return -1;

    }



    avctx->codec_tag = AV_RL32((const uint8_t*)profiles[avctx->profile].name);



    for (i = 1; i <= 16; i++) {

        scale_mat(QMAT_LUMA[avctx->profile]  , ctx->qmat_luma[i - 1]  , i);

        scale_mat(QMAT_CHROMA[avctx->profile], ctx->qmat_chroma[i - 1], i);

    }



    avctx->coded_frame = avcodec_alloc_frame();

    avctx->coded_frame->key_frame = 1;

    avctx->coded_frame->pict_type = AV_PICTURE_TYPE_I;



    return 0;

}
