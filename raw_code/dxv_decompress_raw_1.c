static int dxv_decompress_raw(AVCodecContext *avctx)
{
    DXVContext *ctx = avctx->priv_data;
    GetByteContext *gbc = &ctx->gbc;
    bytestream2_get_buffer(gbc, ctx->tex_data, ctx->tex_size);
    return 0;
}