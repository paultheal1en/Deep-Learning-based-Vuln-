static int dnxhd_decode_header(DNXHDContext *ctx, AVFrame *frame,

                               const uint8_t *buf, int buf_size,

                               int first_field)

{

    static const uint8_t header_prefix[]    = { 0x00, 0x00, 0x02, 0x80, 0x01 };

    static const uint8_t header_prefix444[] = { 0x00, 0x00, 0x02, 0x80, 0x02 };

    int i, cid, ret;



    if (buf_size < 0x280)

        return AVERROR_INVALIDDATA;



    if (memcmp(buf, header_prefix, 5) && memcmp(buf, header_prefix444, 5)) {

        av_log(ctx->avctx, AV_LOG_ERROR, "error in header\n");

        return AVERROR_INVALIDDATA;

    }

    if (buf[5] & 2) { /* interlaced */

        ctx->cur_field = buf[5] & 1;

        frame->interlaced_frame = 1;

        frame->top_field_first  = first_field ^ ctx->cur_field;

        av_log(ctx->avctx, AV_LOG_DEBUG,

               "interlaced %d, cur field %d\n", buf[5] & 3, ctx->cur_field);

    }



    ctx->height = AV_RB16(buf + 0x18);

    ctx->width  = AV_RB16(buf + 0x1a);



    av_dlog(ctx->avctx, "width %d, height %d\n", ctx->width, ctx->height);



    ctx->is_444 = 0;

    if (buf[0x4] == 0x2) {

        ctx->avctx->pix_fmt = AV_PIX_FMT_YUV444P10;

        ctx->avctx->bits_per_raw_sample = 10;

        if (ctx->bit_depth != 10) {

            ff_blockdsp_init(&ctx->bdsp, ctx->avctx);

            ff_idctdsp_init(&ctx->idsp, ctx->avctx);

            ctx->bit_depth = 10;

            ctx->decode_dct_block = dnxhd_decode_dct_block_10_444;

        }

        ctx->is_444 = 1;

    } else if (buf[0x21] & 0x40) {

        ctx->avctx->pix_fmt = AV_PIX_FMT_YUV422P10;

        ctx->avctx->bits_per_raw_sample = 10;

        if (ctx->bit_depth != 10) {

            ff_blockdsp_init(&ctx->bdsp, ctx->avctx);

            ff_idctdsp_init(&ctx->idsp, ctx->avctx);

            ctx->bit_depth = 10;

            ctx->decode_dct_block = dnxhd_decode_dct_block_10;

        }

    } else {

        ctx->avctx->pix_fmt = AV_PIX_FMT_YUV422P;

        ctx->avctx->bits_per_raw_sample = 8;

        if (ctx->bit_depth != 8) {

            ff_blockdsp_init(&ctx->bdsp, ctx->avctx);

            ff_idctdsp_init(&ctx->idsp, ctx->avctx);

            ctx->bit_depth = 8;

            ctx->decode_dct_block = dnxhd_decode_dct_block_8;

        }

    }



    cid = AV_RB32(buf + 0x28);

    av_dlog(ctx->avctx, "compression id %d\n", cid);



    if ((ret = dnxhd_init_vlc(ctx, cid)) < 0)

        return ret;



    if (buf_size < ctx->cid_table->coding_unit_size) {

        av_log(ctx->avctx, AV_LOG_ERROR, "incorrect frame size\n");

        return AVERROR_INVALIDDATA;

    }



    ctx->mb_width  = ctx->width >> 4;

    ctx->mb_height = buf[0x16d];



    av_dlog(ctx->avctx,

            "mb width %d, mb height %d\n", ctx->mb_width, ctx->mb_height);



    if ((ctx->height + 15) >> 4 == ctx->mb_height && frame->interlaced_frame)

        ctx->height <<= 1;



    if (ctx->mb_height > 68 ||

        (ctx->mb_height << frame->interlaced_frame) > (ctx->height + 15) >> 4) {

        av_log(ctx->avctx, AV_LOG_ERROR,

               "mb height too big: %d\n", ctx->mb_height);

        return AVERROR_INVALIDDATA;

    }



    for (i = 0; i < ctx->mb_height; i++) {

        ctx->mb_scan_index[i] = AV_RB32(buf + 0x170 + (i << 2));

        av_dlog(ctx->avctx, "mb scan index %d\n", ctx->mb_scan_index[i]);

        if (buf_size < ctx->mb_scan_index[i] + 0x280LL) {

            av_log(ctx->avctx, AV_LOG_ERROR, "invalid mb scan index\n");

            return AVERROR_INVALIDDATA;

        }

    }



    return 0;

}
