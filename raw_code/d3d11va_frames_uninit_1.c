static void d3d11va_frames_uninit(AVHWFramesContext *ctx)

{

    AVD3D11VAFramesContext *frames_hwctx = ctx->hwctx;

    D3D11VAFramesContext *s = ctx->internal->priv;



    if (frames_hwctx->texture)

        ID3D11Texture2D_Release(frames_hwctx->texture);




    if (s->staging_texture)

        ID3D11Texture2D_Release(s->staging_texture);


}