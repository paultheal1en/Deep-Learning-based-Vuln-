static av_cold int libx265_encode_init(AVCodecContext *avctx)

{

    libx265Context *ctx = avctx->priv_data;



    ctx->api = x265_api_get(av_pix_fmt_desc_get(avctx->pix_fmt)->comp[0].depth);

    if (!ctx->api)

        ctx->api = x265_api_get(0);



    ctx->params = ctx->api->param_alloc();

    if (!ctx->params) {

        av_log(avctx, AV_LOG_ERROR, "Could not allocate x265 param structure.\n");

        return AVERROR(ENOMEM);




    if (ctx->api->param_default_preset(ctx->params, ctx->preset, ctx->tune) < 0) {

        int i;



        av_log(avctx, AV_LOG_ERROR, "Error setting preset/tune %s/%s.\n", ctx->preset, ctx->tune);

        av_log(avctx, AV_LOG_INFO, "Possible presets:");

        for (i = 0; x265_preset_names[i]; i++)

            av_log(avctx, AV_LOG_INFO, " %s", x265_preset_names[i]);



        av_log(avctx, AV_LOG_INFO, "\n");

        av_log(avctx, AV_LOG_INFO, "Possible tunes:");

        for (i = 0; x265_tune_names[i]; i++)

            av_log(avctx, AV_LOG_INFO, " %s", x265_tune_names[i]);



        av_log(avctx, AV_LOG_INFO, "\n");



        return AVERROR(EINVAL);




    ctx->params->frameNumThreads = avctx->thread_count;

    ctx->params->fpsNum          = avctx->time_base.den;

    ctx->params->fpsDenom        = avctx->time_base.num * avctx->ticks_per_frame;

    ctx->params->sourceWidth     = avctx->width;

    ctx->params->sourceHeight    = avctx->height;

    ctx->params->bEnablePsnr     = !!(avctx->flags & AV_CODEC_FLAG_PSNR);



    if ((avctx->color_primaries <= AVCOL_PRI_BT2020 &&

         avctx->color_primaries != AVCOL_PRI_UNSPECIFIED) ||

        (avctx->color_trc <= AVCOL_TRC_BT2020_12 &&

         avctx->color_trc != AVCOL_TRC_UNSPECIFIED) ||

        (avctx->colorspace <= AVCOL_SPC_BT2020_CL &&

         avctx->colorspace != AVCOL_SPC_UNSPECIFIED)) {



        ctx->params->vui.bEnableVideoSignalTypePresentFlag  = 1;

        ctx->params->vui.bEnableColorDescriptionPresentFlag = 1;



        // x265 validates the parameters internally

        ctx->params->vui.colorPrimaries          = avctx->color_primaries;

        ctx->params->vui.transferCharacteristics = avctx->color_trc;

        ctx->params->vui.matrixCoeffs            = avctx->colorspace;




    if (avctx->sample_aspect_ratio.num > 0 && avctx->sample_aspect_ratio.den > 0) {

        char sar[12];

        int sar_num, sar_den;



        av_reduce(&sar_num, &sar_den,

                  avctx->sample_aspect_ratio.num,

                  avctx->sample_aspect_ratio.den, 65535);

        snprintf(sar, sizeof(sar), "%d:%d", sar_num, sar_den);

        if (ctx->api->param_parse(ctx->params, "sar", sar) == X265_PARAM_BAD_VALUE) {

            av_log(avctx, AV_LOG_ERROR, "Invalid SAR: %d:%d.\n", sar_num, sar_den);






    switch (avctx->pix_fmt) {

    case AV_PIX_FMT_YUV420P:

    case AV_PIX_FMT_YUV420P10:

    case AV_PIX_FMT_YUV420P12:

        ctx->params->internalCsp = X265_CSP_I420;


    case AV_PIX_FMT_YUV422P:

    case AV_PIX_FMT_YUV422P10:

    case AV_PIX_FMT_YUV422P12:

        ctx->params->internalCsp = X265_CSP_I422;


    case AV_PIX_FMT_GBRP:

    case AV_PIX_FMT_GBRP10:

    case AV_PIX_FMT_GBRP12:

        ctx->params->vui.matrixCoeffs = AVCOL_SPC_RGB;

        ctx->params->vui.bEnableVideoSignalTypePresentFlag  = 1;

        ctx->params->vui.bEnableColorDescriptionPresentFlag = 1;

    case AV_PIX_FMT_YUV444P:

    case AV_PIX_FMT_YUV444P10:

    case AV_PIX_FMT_YUV444P12:

        ctx->params->internalCsp = X265_CSP_I444;














    if (ctx->crf >= 0) {

        char crf[6];



        snprintf(crf, sizeof(crf), "%2.2f", ctx->crf);

        if (ctx->api->param_parse(ctx->params, "crf", crf) == X265_PARAM_BAD_VALUE) {

            av_log(avctx, AV_LOG_ERROR, "Invalid crf: %2.2f.\n", ctx->crf);

            return AVERROR(EINVAL);


    } else if (avctx->bit_rate > 0) {

        ctx->params->rc.bitrate         = avctx->bit_rate / 1000;

        ctx->params->rc.rateControlMode = X265_RC_ABR;




    if (!(avctx->flags & AV_CODEC_FLAG_GLOBAL_HEADER))

        ctx->params->bRepeatHeaders = 1;



    if (ctx->x265_opts) {

        AVDictionary *dict    = NULL;

        AVDictionaryEntry *en = NULL;



        if (!av_dict_parse_string(&dict, ctx->x265_opts, "=", ":", 0)) {

            while ((en = av_dict_get(dict, "", en, AV_DICT_IGNORE_SUFFIX))) {

                int parse_ret = ctx->api->param_parse(ctx->params, en->key, en->value);



                switch (parse_ret) {

                case X265_PARAM_BAD_NAME:

                    av_log(avctx, AV_LOG_WARNING,

                          "Unknown option: %s.\n", en->key);


                case X265_PARAM_BAD_VALUE:

                    av_log(avctx, AV_LOG_WARNING,

                          "Invalid value for %s: %s.\n", en->key, en->value);


                default:




            av_dict_free(&dict);





    ctx->encoder = ctx->api->encoder_open(ctx->params);

    if (!ctx->encoder) {

        av_log(avctx, AV_LOG_ERROR, "Cannot open libx265 encoder.\n");

        libx265_encode_close(avctx);





    if (avctx->flags & AV_CODEC_FLAG_GLOBAL_HEADER) {

        x265_nal *nal;

        int nnal;



        avctx->extradata_size = ctx->api->encoder_headers(ctx->encoder, &nal, &nnal);

        if (avctx->extradata_size <= 0) {

            av_log(avctx, AV_LOG_ERROR, "Cannot encode headers.\n");

            libx265_encode_close(avctx);





        avctx->extradata = av_malloc(avctx->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);

        if (!avctx->extradata) {


                   "Cannot allocate HEVC header of size %d.\n", avctx->extradata_size);

            libx265_encode_close(avctx);

            return AVERROR(ENOMEM);




        memcpy(avctx->extradata, nal[0].payload, avctx->extradata_size);




    return 0;
