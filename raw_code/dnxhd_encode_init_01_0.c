static int dnxhd_encode_init(AVCodecContext *avctx)

{

    DNXHDEncContext *ctx = avctx->priv_data;

    int i, index, bit_depth;



    switch (avctx->pix_fmt) {

    case AV_PIX_FMT_YUV422P:

        bit_depth = 8;

        break;

    case AV_PIX_FMT_YUV422P10:

        bit_depth = 10;

        break;

    default:

        av_log(avctx, AV_LOG_ERROR, "pixel format is incompatible with DNxHD\n");

        return -1;

    }



    ctx->cid = ff_dnxhd_find_cid(avctx, bit_depth);

    if (!ctx->cid) {

        av_log(avctx, AV_LOG_ERROR, "video parameters incompatible with DNxHD\n");

        return -1;

    }

    av_log(avctx, AV_LOG_DEBUG, "cid %d\n", ctx->cid);



    index = ff_dnxhd_get_cid_table(ctx->cid);

    av_assert0(index >= 0);

    ctx->cid_table = &ff_dnxhd_cid_table[index];



    ctx->m.avctx = avctx;

    ctx->m.mb_intra = 1;

    ctx->m.h263_aic = 1;



    avctx->bits_per_raw_sample = ctx->cid_table->bit_depth;



    ff_dct_common_init(&ctx->m);

    ff_dct_encode_init(&ctx->m);



    if (!ctx->m.dct_quantize)

        ctx->m.dct_quantize = ff_dct_quantize_c;



    if (ctx->cid_table->bit_depth == 10) {

       ctx->m.dct_quantize = dnxhd_10bit_dct_quantize;

       ctx->get_pixels_8x4_sym = dnxhd_10bit_get_pixels_8x4_sym;

       ctx->block_width_l2 = 4;

    } else {

       ctx->get_pixels_8x4_sym = dnxhd_8bit_get_pixels_8x4_sym;

       ctx->block_width_l2 = 3;

    }



    if (ARCH_X86)

        ff_dnxhdenc_init_x86(ctx);



    ctx->m.mb_height = (avctx->height + 15) / 16;

    ctx->m.mb_width  = (avctx->width  + 15) / 16;



    if (avctx->flags & CODEC_FLAG_INTERLACED_DCT) {

        ctx->interlaced = 1;

        ctx->m.mb_height /= 2;

    }



    ctx->m.mb_num = ctx->m.mb_height * ctx->m.mb_width;



    if (avctx->intra_quant_bias != FF_DEFAULT_QUANT_BIAS)

        ctx->m.intra_quant_bias = avctx->intra_quant_bias;

    if (dnxhd_init_qmat(ctx, ctx->m.intra_quant_bias, 0) < 0) // XXX tune lbias/cbias

        return -1;



    // Avid Nitris hardware decoder requires a minimum amount of padding in the coding unit payload

    if (ctx->nitris_compat)

        ctx->min_padding = 1600;



    if (dnxhd_init_vlc(ctx) < 0)

        return -1;

    if (dnxhd_init_rc(ctx) < 0)

        return -1;



    FF_ALLOCZ_OR_GOTO(ctx->m.avctx, ctx->slice_size, ctx->m.mb_height*sizeof(uint32_t), fail);

    FF_ALLOCZ_OR_GOTO(ctx->m.avctx, ctx->slice_offs, ctx->m.mb_height*sizeof(uint32_t), fail);

    FF_ALLOCZ_OR_GOTO(ctx->m.avctx, ctx->mb_bits,    ctx->m.mb_num   *sizeof(uint16_t), fail);

    FF_ALLOCZ_OR_GOTO(ctx->m.avctx, ctx->mb_qscale,  ctx->m.mb_num   *sizeof(uint8_t),  fail);



    ctx->frame.key_frame = 1;

    ctx->frame.pict_type = AV_PICTURE_TYPE_I;

    ctx->m.avctx->coded_frame = &ctx->frame;



    if (avctx->thread_count > MAX_THREADS) {

        av_log(avctx, AV_LOG_ERROR, "too many threads\n");

        return -1;

    }



    ctx->thread[0] = ctx;

    for (i = 1; i < avctx->thread_count; i++) {

        ctx->thread[i] =  av_malloc(sizeof(DNXHDEncContext));

        memcpy(ctx->thread[i], ctx, sizeof(DNXHDEncContext));

    }



    return 0;

 fail: //for FF_ALLOCZ_OR_GOTO

    return -1;

}
