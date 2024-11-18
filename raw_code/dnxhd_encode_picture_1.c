static int dnxhd_encode_picture(AVCodecContext *avctx, unsigned char *buf, int buf_size, void *data)
{
    DNXHDEncContext *ctx = avctx->priv_data;
    int first_field = 1;
    int offset, i, ret;
    if (buf_size < ctx->cid_table->frame_size) {
        av_log(avctx, AV_LOG_ERROR, "output buffer is too small to compress picture\n");
        return -1;
    }
    dnxhd_load_picture(ctx, data);
 encode_coding_unit:
    for (i = 0; i < 3; i++) {
        ctx->src[i] = ctx->frame.data[i];
        if (ctx->interlaced && ctx->cur_field)
            ctx->src[i] += ctx->frame.linesize[i];
    }
    dnxhd_write_header(avctx, buf);
    if (avctx->mb_decision == FF_MB_DECISION_RD)
        ret = dnxhd_encode_rdo(avctx, ctx);
    else
        ret = dnxhd_encode_fast(avctx, ctx);
    if (ret < 0) {
        av_log(avctx, AV_LOG_ERROR, "picture could not fit ratecontrol constraints\n");
        return -1;
    }
    dnxhd_setup_threads_slices(ctx, buf);
    offset = 0;
    for (i = 0; i < ctx->m.mb_height; i++) {
        AV_WB32(ctx->msip + i * 4, offset);
        offset += ctx->slice_size[i];
        assert(!(ctx->slice_size[i] & 3));
    }
    avctx->execute(avctx, dnxhd_encode_thread, (void**)&ctx->thread[0], NULL, avctx->thread_count, sizeof(void*));
    AV_WB32(buf + ctx->cid_table->coding_unit_size - 4, 0x600DC0DE); // EOF
    if (ctx->interlaced && first_field) {
        first_field     = 0;
        ctx->cur_field ^= 1;
        buf      += ctx->cid_table->coding_unit_size;
        buf_size -= ctx->cid_table->coding_unit_size;
        goto encode_coding_unit;
    }
    ctx->frame.quality = ctx->qscale*FF_QP2LAMBDA;
    return ctx->cid_table->frame_size;
}