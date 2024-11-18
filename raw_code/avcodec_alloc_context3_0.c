AVCodecContext *avcodec_alloc_context3(const AVCodec *codec)

{

    AVCodecContext *avctx= av_malloc(sizeof(AVCodecContext));



    if(avctx==NULL) return NULL;



    if(avcodec_get_context_defaults3(avctx, codec) < 0){

        av_free(avctx);

        return NULL;

    }



    return avctx;

}
