int avcodec_decode_audio(AVCodecContext *avctx, int16_t *samples, 

                         int *frame_size_ptr,

                         uint8_t *buf, int buf_size)

{

    int ret;



    *frame_size_ptr= 0;

    ret = avctx->codec->decode(avctx, samples, frame_size_ptr, 

                               buf, buf_size);

    avctx->frame_number++;

    return ret;

}
