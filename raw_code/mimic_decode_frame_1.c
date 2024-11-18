static int mimic_decode_frame(AVCodecContext *avctx, void *data,

                              int *got_frame, AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size       = avpkt->size;

    int swap_buf_size  = buf_size - MIMIC_HEADER_SIZE;

    MimicContext *ctx  = avctx->priv_data;

    GetByteContext gb;

    int is_pframe;

    int width, height;

    int quality, num_coeffs;

    int res;



    if (buf_size <= MIMIC_HEADER_SIZE) {

        av_log(avctx, AV_LOG_ERROR, "insufficient data\n");

        return AVERROR_INVALIDDATA;

    }



    bytestream2_init(&gb, buf, MIMIC_HEADER_SIZE);

    bytestream2_skip(&gb, 2); /* some constant (always 256) */

    quality    = bytestream2_get_le16u(&gb);

    width      = bytestream2_get_le16u(&gb);

    height     = bytestream2_get_le16u(&gb);

    bytestream2_skip(&gb, 4); /* some constant */

    is_pframe  = bytestream2_get_le32u(&gb);

    num_coeffs = bytestream2_get_byteu(&gb);

    bytestream2_skip(&gb, 3); /* some constant */



    if (!ctx->avctx) {

        int i;



        if (!(width == 160 && height == 120) &&

            !(width == 320 && height == 240)) {

            av_log(avctx, AV_LOG_ERROR, "invalid width/height!\n");

            return AVERROR_INVALIDDATA;

        }



        ctx->avctx     = avctx;

        avctx->width   = width;

        avctx->height  = height;

        avctx->pix_fmt = AV_PIX_FMT_YUV420P;

        for (i = 0; i < 3; i++) {

            ctx->num_vblocks[i] = AV_CEIL_RSHIFT(height,   3 + !!i);

            ctx->num_hblocks[i] =                width >> (3 + !!i);

        }

    } else if (width != ctx->avctx->width || height != ctx->avctx->height) {

        avpriv_request_sample(avctx, "Resolution changing");

        return AVERROR_PATCHWELCOME;

    }



    if (is_pframe && !ctx->frames[ctx->prev_index].f->data[0]) {

        av_log(avctx, AV_LOG_ERROR, "decoding must start with keyframe\n");

        return AVERROR_INVALIDDATA;

    }



    ff_thread_release_buffer(avctx, &ctx->frames[ctx->cur_index]);

    ctx->frames[ctx->cur_index].f->pict_type = is_pframe ? AV_PICTURE_TYPE_P :

                                                           AV_PICTURE_TYPE_I;

    if ((res = ff_thread_get_buffer(avctx, &ctx->frames[ctx->cur_index],

                                    AV_GET_BUFFER_FLAG_REF)) < 0) {

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return res;

    }



    ctx->next_prev_index = ctx->cur_index;

    ctx->next_cur_index  = (ctx->cur_index - 1) & 15;



    ff_thread_finish_setup(avctx);



    av_fast_padded_malloc(&ctx->swap_buf, &ctx->swap_buf_size, swap_buf_size);

    if (!ctx->swap_buf)

        return AVERROR(ENOMEM);



    ctx->bbdsp.bswap_buf(ctx->swap_buf,

                         (const uint32_t *) (buf + MIMIC_HEADER_SIZE),

                         swap_buf_size >> 2);

    init_get_bits(&ctx->gb, ctx->swap_buf, swap_buf_size << 3);



    res = decode(ctx, quality, num_coeffs, !is_pframe);

    ff_thread_report_progress(&ctx->frames[ctx->cur_index], INT_MAX, 0);

    if (res < 0) {

        if (!(avctx->active_thread_type & FF_THREAD_FRAME))

            ff_thread_release_buffer(avctx, &ctx->frames[ctx->cur_index]);

        return res;

    }



    if ((res = av_frame_ref(data, ctx->frames[ctx->cur_index].f)) < 0)

        return res;

    *got_frame      = 1;



    flip_swap_frame(data);



    ctx->prev_index = ctx->next_prev_index;

    ctx->cur_index  = ctx->next_cur_index;



    /* Only release frames that aren't used for backreferences anymore */

    ff_thread_release_buffer(avctx, &ctx->frames[ctx->cur_index]);



    return buf_size;

}
