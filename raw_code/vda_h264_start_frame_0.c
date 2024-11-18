static int vda_h264_start_frame(AVCodecContext *avctx,

                                av_unused const uint8_t *buffer,

                                av_unused uint32_t size)

{

    VDAContext *vda = avctx->internal->hwaccel_priv_data;

    struct vda_context *vda_ctx         = avctx->hwaccel_context;



    if (!vda_ctx->decoder)

        return -1;



    vda->bitstream_size = 0;



    return 0;

}
