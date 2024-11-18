static av_cold int libopenjpeg_encode_init(AVCodecContext *avctx)

{

    LibOpenJPEGContext *ctx = avctx->priv_data;

    int err = 0;



    opj_set_default_encoder_parameters(&ctx->enc_params);



#if HAVE_OPENJPEG_2_1_OPENJPEG_H

    switch (ctx->cinema_mode) {

    case OPJ_CINEMA2K_24:

        ctx->enc_params.rsiz = OPJ_PROFILE_CINEMA_2K;

        ctx->enc_params.max_cs_size = OPJ_CINEMA_24_CS;

        ctx->enc_params.max_comp_size = OPJ_CINEMA_24_COMP;

        break;

    case OPJ_CINEMA2K_48:

        ctx->enc_params.rsiz = OPJ_PROFILE_CINEMA_2K;

        ctx->enc_params.max_cs_size = OPJ_CINEMA_48_CS;

        ctx->enc_params.max_comp_size = OPJ_CINEMA_48_COMP;

        break;

    case OPJ_CINEMA4K_24:

        ctx->enc_params.rsiz = OPJ_PROFILE_CINEMA_4K;

        ctx->enc_params.max_cs_size = OPJ_CINEMA_24_CS;

        ctx->enc_params.max_comp_size = OPJ_CINEMA_24_COMP;

        break;

    }



    switch (ctx->profile) {

    case OPJ_CINEMA2K:

        if (ctx->enc_params.rsiz == OPJ_PROFILE_CINEMA_4K) {

            err = AVERROR(EINVAL);

            break;

        }

        ctx->enc_params.rsiz = OPJ_PROFILE_CINEMA_2K;

        break;

    case OPJ_CINEMA4K:

        if (ctx->enc_params.rsiz == OPJ_PROFILE_CINEMA_2K) {

            err = AVERROR(EINVAL);

            break;

        }

        ctx->enc_params.rsiz = OPJ_PROFILE_CINEMA_4K;

        break;

    }



    if (err) {

        av_log(avctx, AV_LOG_ERROR,

               "Invalid parameter pairing: cinema_mode and profile conflict.\n");

        goto fail;

    }

#else

    ctx->enc_params.cp_rsiz = ctx->profile;

    ctx->enc_params.cp_cinema = ctx->cinema_mode;

#endif



    if (!ctx->numresolution) {

        ctx->numresolution = 6;

        while (FFMIN(avctx->width, avctx->height) >> ctx->numresolution < 1)

            ctx->numresolution --;

    }



    ctx->enc_params.mode = !!avctx->global_quality;

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



#if OPENJPEG_MAJOR_VERSION == 1

    ctx->image = mj2_create_image(avctx, &ctx->enc_params);

    if (!ctx->image) {

        av_log(avctx, AV_LOG_ERROR, "Error creating the mj2 image\n");

        err = AVERROR(EINVAL);

        goto fail;

    }

#endif // OPENJPEG_MAJOR_VERSION == 1



    return 0;



fail:

#if OPENJPEG_MAJOR_VERSION == 1

    opj_image_destroy(ctx->image);

    ctx->image = NULL;

#endif // OPENJPEG_MAJOR_VERSION == 1

    return err;

}
