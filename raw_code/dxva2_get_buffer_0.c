static int dxva2_get_buffer(AVCodecContext *s, AVFrame *frame, int flags)

{

    InputStream  *ist = s->opaque;

    DXVA2Context *ctx = ist->hwaccel_ctx;



    return av_hwframe_get_buffer(ctx->hw_frames_ctx, frame, 0);

}
