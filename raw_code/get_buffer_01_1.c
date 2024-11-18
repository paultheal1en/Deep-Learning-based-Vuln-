static int get_buffer(AVCodecContext *avctx, AVFrame *pic)

{

    pic->type = FF_BUFFER_TYPE_USER;

    pic->data[0] = (void *)1;

    return 0;

}
