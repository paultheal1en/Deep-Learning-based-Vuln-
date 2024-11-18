static av_cold int vaapi_encode_init_rate_control(AVCodecContext *avctx)

{

    VAAPIEncodeContext *ctx = avctx->priv_data;

    int hrd_buffer_size;

    int hrd_initial_buffer_fullness;



    if (avctx->rc_buffer_size)

        hrd_buffer_size = avctx->rc_buffer_size;

    else

        hrd_buffer_size = avctx->bit_rate;

    if (avctx->rc_initial_buffer_occupancy)

        hrd_initial_buffer_fullness = avctx->rc_initial_buffer_occupancy;

    else

        hrd_initial_buffer_fullness = hrd_buffer_size * 3 / 4;



    ctx->rc_params.misc.type = VAEncMiscParameterTypeRateControl;

    ctx->rc_params.rc = (VAEncMiscParameterRateControl) {

        .bits_per_second   = avctx->bit_rate,

        .target_percentage = 66,

        .window_size       = 1000,

        .initial_qp        = (avctx->qmax >= 0 ? avctx->qmax : 40),

        .min_qp            = (avctx->qmin >= 0 ? avctx->qmin : 18),

        .basic_unit_size   = 0,

    };

    ctx->global_params[ctx->nb_global_params] =

        &ctx->rc_params.misc;

    ctx->global_params_size[ctx->nb_global_params++] =

        sizeof(ctx->rc_params);



    ctx->hrd_params.misc.type = VAEncMiscParameterTypeHRD;

    ctx->hrd_params.hrd = (VAEncMiscParameterHRD) {

        .initial_buffer_fullness = hrd_initial_buffer_fullness,

        .buffer_size             = hrd_buffer_size,

    };

    ctx->global_params[ctx->nb_global_params] =

        &ctx->hrd_params.misc;

    ctx->global_params_size[ctx->nb_global_params++] =

        sizeof(ctx->hrd_params);



    return 0;

}
