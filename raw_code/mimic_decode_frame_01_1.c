static int mimic_decode_frame(AVCodecContext *avctx, void *data,

                              int *data_size, AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    MimicContext *ctx = avctx->priv_data;

    GetByteContext gb;

    int is_pframe;

    int width, height;

    int quality, num_coeffs;

    int swap_buf_size = buf_size - MIMIC_HEADER_SIZE;



    if (buf_size <= MIMIC_HEADER_SIZE) {

        av_log(avctx, AV_LOG_ERROR, "insufficient data\n");

        return -1;

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



    if(!ctx->avctx) {

        int i;



        if(!(width == 160 && height == 120) &&

           !(width == 320 && height == 240)) {

            av_log(avctx, AV_LOG_ERROR, "invalid width/height!\n");

            return -1;

        }



        ctx->avctx     = avctx;

        avctx->width   = width;

        avctx->height  = height;

        avctx->pix_fmt = PIX_FMT_YUV420P;

        for(i = 0; i < 3; i++) {

            ctx->num_vblocks[i] = -((-height) >> (3 + !!i));

            ctx->num_hblocks[i] =     width   >> (3 + !!i) ;

        }

    } else if(width != ctx->avctx->width || height != ctx->avctx->height) {

        av_log(avctx, AV_LOG_ERROR, "resolution changing is not supported\n");

        return -1;

    }



    if(is_pframe && !ctx->buf_ptrs[ctx->prev_index].data[0]) {

        av_log(avctx, AV_LOG_ERROR, "decoding must start with keyframe\n");

        return -1;

    }



    ctx->buf_ptrs[ctx->cur_index].reference = 1;

    ctx->buf_ptrs[ctx->cur_index].pict_type = is_pframe ? AV_PICTURE_TYPE_P:AV_PICTURE_TYPE_I;

    if(ff_thread_get_buffer(avctx, &ctx->buf_ptrs[ctx->cur_index])) {

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return -1;

    }



    ctx->next_prev_index = ctx->cur_index;

    ctx->next_cur_index  = (ctx->cur_index - 1) & 15;



    prepare_avpic(ctx, &ctx->flipped_ptrs[ctx->cur_index],

                  (AVPicture*) &ctx->buf_ptrs[ctx->cur_index]);



    ff_thread_finish_setup(avctx);



    av_fast_malloc(&ctx->swap_buf, &ctx->swap_buf_size,

                                 swap_buf_size + FF_INPUT_BUFFER_PADDING_SIZE);

    if(!ctx->swap_buf)

        return AVERROR(ENOMEM);



    ctx->dsp.bswap_buf(ctx->swap_buf,

                        (const uint32_t*) (buf + MIMIC_HEADER_SIZE),

                        swap_buf_size>>2);

    init_get_bits(&ctx->gb, ctx->swap_buf, swap_buf_size << 3);



    if(!decode(ctx, quality, num_coeffs, !is_pframe)) {

        if (avctx->active_thread_type&FF_THREAD_FRAME)

            ff_thread_report_progress(&ctx->buf_ptrs[ctx->cur_index], INT_MAX, 0);

        else {

            ff_thread_release_buffer(avctx, &ctx->buf_ptrs[ctx->cur_index]);

            return -1;

        }

    }



    *(AVFrame*)data = ctx->buf_ptrs[ctx->cur_index];

    *data_size = sizeof(AVFrame);



    ctx->prev_index = ctx->next_prev_index;

    ctx->cur_index  = ctx->next_cur_index;



    /* Only release frames that aren't used for backreferences anymore */

    if(ctx->buf_ptrs[ctx->cur_index].data[0])

        ff_thread_release_buffer(avctx, &ctx->buf_ptrs[ctx->cur_index]);



    return buf_size;

}
