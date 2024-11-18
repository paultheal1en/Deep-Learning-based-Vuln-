static void vda_h264_uninit(AVCodecContext *avctx)

{

    VDAContext *vda = avctx->internal->priv_data;

    av_freep(&vda->bitstream);

}
