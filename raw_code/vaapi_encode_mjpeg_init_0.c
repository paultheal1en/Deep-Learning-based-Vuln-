static av_cold int vaapi_encode_mjpeg_init(AVCodecContext *avctx)

{

    return ff_vaapi_encode_init(avctx, &vaapi_encode_type_mjpeg);

}
