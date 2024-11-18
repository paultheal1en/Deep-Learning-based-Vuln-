void ff_thread_release_buffer(AVCodecContext *avctx, ThreadFrame *f)

{

    PerThreadContext *p = avctx->internal->thread_ctx;

    FrameThreadContext *fctx;

    AVFrame *dst, *tmp;

FF_DISABLE_DEPRECATION_WARNINGS

    int can_direct_free = !(avctx->active_thread_type & FF_THREAD_FRAME) ||

                          avctx->thread_safe_callbacks                   ||

                          (

#if FF_API_GET_BUFFER

                           !avctx->get_buffer &&

#endif

                           avctx->get_buffer2 == avcodec_default_get_buffer2);

FF_ENABLE_DEPRECATION_WARNINGS



    if (!f->f->buf[0])

        return;



    if (avctx->debug & FF_DEBUG_BUFFERS)

        av_log(avctx, AV_LOG_DEBUG, "thread_release_buffer called on pic %p\n", f);



    av_buffer_unref(&f->progress);

    f->owner    = NULL;



    if (can_direct_free) {

        av_frame_unref(f->f);

        return;

    }



    fctx = p->parent;

    pthread_mutex_lock(&fctx->buffer_mutex);



    if (p->num_released_buffers + 1 >= INT_MAX / sizeof(*p->released_buffers))

        goto fail;

    tmp = av_fast_realloc(p->released_buffers, &p->released_buffers_allocated,

                          (p->num_released_buffers + 1) *

                          sizeof(*p->released_buffers));

    if (!tmp)

        goto fail;

    p->released_buffers = tmp;



    dst = &p->released_buffers[p->num_released_buffers];

    av_frame_move_ref(dst, f->f);



    p->num_released_buffers++;



fail:

    pthread_mutex_unlock(&fctx->buffer_mutex);

}
