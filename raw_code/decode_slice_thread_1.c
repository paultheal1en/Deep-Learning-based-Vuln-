static int decode_slice_thread(AVCodecContext *avctx, void *arg, int jobnr, int threadnr)

{

    ProresContext *ctx = avctx->priv_data;

    SliceContext *slice = &ctx->slices[jobnr];

    const uint8_t *buf = slice->data;

    AVFrame *pic = ctx->frame;

    int i, hdr_size, qscale, log2_chroma_blocks_per_mb;

    int luma_stride, chroma_stride;

    int y_data_size, u_data_size, v_data_size, a_data_size;

    uint8_t *dest_y, *dest_u, *dest_v, *dest_a;

    int16_t qmat_luma_scaled[64];

    int16_t qmat_chroma_scaled[64];

    int mb_x_shift;



    slice->ret = -1;

    //av_log(avctx, AV_LOG_INFO, "slice %d mb width %d mb x %d y %d\n",

    //       jobnr, slice->mb_count, slice->mb_x, slice->mb_y);



    // slice header

    hdr_size = buf[0] >> 3;

    qscale = av_clip(buf[1], 1, 224);

    qscale = qscale > 128 ? qscale - 96 << 2: qscale;

    y_data_size = AV_RB16(buf + 2);

    u_data_size = AV_RB16(buf + 4);

    v_data_size = slice->data_size - y_data_size - u_data_size - hdr_size;

    if (hdr_size > 7) v_data_size = AV_RB16(buf + 6);

    a_data_size = slice->data_size - y_data_size - u_data_size -

                  v_data_size - hdr_size;



    if (y_data_size < 0 || u_data_size < 0 || v_data_size < 0

        || hdr_size+y_data_size+u_data_size+v_data_size > slice->data_size){

        av_log(avctx, AV_LOG_ERROR, "invalid plane data size\n");

        return -1;

    }



    buf += hdr_size;



    for (i = 0; i < 64; i++) {

        qmat_luma_scaled  [i] = ctx->qmat_luma  [i] * qscale;

        qmat_chroma_scaled[i] = ctx->qmat_chroma[i] * qscale;

    }



    if (ctx->frame_type == 0) {

        luma_stride   = pic->linesize[0];

        chroma_stride = pic->linesize[1];

    } else {

        luma_stride   = pic->linesize[0] << 1;

        chroma_stride = pic->linesize[1] << 1;

    }



    if (avctx->pix_fmt == AV_PIX_FMT_YUV444P10 || avctx->pix_fmt == AV_PIX_FMT_YUVA444P10) {

        mb_x_shift = 5;

        log2_chroma_blocks_per_mb = 2;

    } else {

        mb_x_shift = 4;

        log2_chroma_blocks_per_mb = 1;

    }



    dest_y = pic->data[0] + (slice->mb_y << 4) * luma_stride + (slice->mb_x << 5);

    dest_u = pic->data[1] + (slice->mb_y << 4) * chroma_stride + (slice->mb_x << mb_x_shift);

    dest_v = pic->data[2] + (slice->mb_y << 4) * chroma_stride + (slice->mb_x << mb_x_shift);

    dest_a = pic->data[3] + (slice->mb_y << 4) * luma_stride + (slice->mb_x << 5);



    if (ctx->frame_type && ctx->first_field ^ ctx->frame->top_field_first) {

        dest_y += pic->linesize[0];

        dest_u += pic->linesize[1];

        dest_v += pic->linesize[2];

        dest_a += pic->linesize[3];

    }



    decode_slice_luma(avctx, slice, (uint16_t*)dest_y, luma_stride,

                      buf, y_data_size, qmat_luma_scaled);



    if (!(avctx->flags & CODEC_FLAG_GRAY)) {

        decode_slice_chroma(avctx, slice, (uint16_t*)dest_u, chroma_stride,

                            buf + y_data_size, u_data_size,

                            qmat_chroma_scaled, log2_chroma_blocks_per_mb);

        decode_slice_chroma(avctx, slice, (uint16_t*)dest_v, chroma_stride,

                            buf + y_data_size + u_data_size, v_data_size,

                            qmat_chroma_scaled, log2_chroma_blocks_per_mb);

    }

    /* decode alpha plane if available */

    if (ctx->alpha_info && dest_a && a_data_size)

        decode_slice_alpha(ctx, (uint16_t*)dest_a, luma_stride,

                           buf + y_data_size + u_data_size + v_data_size,

                           a_data_size, slice->mb_count);



    slice->ret = 0;

    return 0;

}
