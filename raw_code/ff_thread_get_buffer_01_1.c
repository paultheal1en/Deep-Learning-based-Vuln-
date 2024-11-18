int ff_thread_get_buffer(AVCodecContext *avctx, ThreadFrame *f, int flags)

{

    f->owner = avctx;

    return ff_get_buffer(avctx, f->f, flags);

}
