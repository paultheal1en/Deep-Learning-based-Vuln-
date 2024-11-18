static av_cold int libopenjpeg_encode_init(AVCodecContext *avctx)

{

    LibOpenJPEGContext *ctx = avctx->priv_data;

    int err = AVERROR(ENOMEM);



    opj_set_default_encoder_parameters(&ctx->enc_params);



    ctx->enc_params.cp_rsiz = ctx->profile;

    ctx->enc_params.mode = !!avctx->global_quality;

    ctx->enc_params.cp_cinema = ctx->cinema_mode;

    ctx->enc_params.prog_order = ctx->prog_order;

    ctx->enc_params.numresolution = ctx->numresolution;

    ctx->enc_params.cp_disto_alloc = ctx->disto_alloc;

    ctx->enc_params.cp_fixed_alloc = ctx->fixed_alloc;

    ctx->enc_params.cp_fixed_quality = ctx->fixed_quality;

    ctx->enc_params.tcp_numlayers = ctx->numlayers;

    ctx->enc_params.tcp_rates[0] = FFMAX(avctx->compression_level, 0) * 2;



    if (ctx->cinema_mode > 0) {

        cinema_parameters(&ctx->enc_params);

    }



    ctx->compress = opj_create_compress(ctx->format);

    if (!ctx->compress) {

        av_log(avctx, AV_LOG_ERROR, "Error creating the compressor\n");

        return AVERROR(ENOMEM);

    }



    ctx->image = mj2_create_image(avctx, &ctx->enc_params);

    if (!ctx->image) {

        av_log(avctx, AV_LOG_ERROR, "Error creating the mj2 image\n");

        err = AVERROR(EINVAL);

        goto fail;

    }



    avctx->coded_frame = av_frame_alloc();

    if (!avctx->coded_frame) {

        av_log(avctx, AV_LOG_ERROR, "Error allocating coded frame\n");

        goto fail;

    }



    memset(&ctx->event_mgr, 0, sizeof(opj_event_mgr_t));

    ctx->event_mgr.info_handler    = info_callback;

    ctx->event_mgr.error_handler = error_callback;

    ctx->event_mgr.warning_handler = warning_callback;

    opj_set_event_mgr((opj_common_ptr) ctx->compress, &ctx->event_mgr, avctx);



    return 0;



fail:

    opj_destroy_compress(ctx->compress);

    ctx->compress = NULL;

    opj_image_destroy(ctx->image);

    ctx->image = NULL;

    av_freep(&avctx->coded_frame);

    return err;

}
