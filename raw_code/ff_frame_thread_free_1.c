void ff_frame_thread_free(AVCodecContext *avctx, int thread_count)

{

    FrameThreadContext *fctx = avctx->internal->thread_ctx;

    const AVCodec *codec = avctx->codec;

    int i;



    park_frame_worker_threads(fctx, thread_count);



    if (fctx->prev_thread && fctx->prev_thread != fctx->threads)

        if (update_context_from_thread(fctx->threads->avctx, fctx->prev_thread->avctx, 0) < 0) {

            av_log(avctx, AV_LOG_ERROR, "Final thread update failed\n");

            fctx->prev_thread->avctx->internal->is_copy = fctx->threads->avctx->internal->is_copy;

            fctx->threads->avctx->internal->is_copy = 1;

        }



    fctx->die = 1;



    for (i = 0; i < thread_count; i++) {

        PerThreadContext *p = &fctx->threads[i];



        pthread_mutex_lock(&p->mutex);

        pthread_cond_signal(&p->input_cond);

        pthread_mutex_unlock(&p->mutex);



        if (p->thread_init)

            pthread_join(p->thread, NULL);

        p->thread_init=0;



        if (codec->close)

            codec->close(p->avctx);



        release_delayed_buffers(p);

        av_frame_free(&p->frame);

    }



    for (i = 0; i < thread_count; i++) {

        PerThreadContext *p = &fctx->threads[i];



        pthread_mutex_destroy(&p->mutex);

        pthread_mutex_destroy(&p->progress_mutex);

        pthread_cond_destroy(&p->input_cond);

        pthread_cond_destroy(&p->progress_cond);

        pthread_cond_destroy(&p->output_cond);

        av_packet_unref(&p->avpkt);

        av_freep(&p->released_buffers);



        if (i) {

            av_freep(&p->avctx->priv_data);

            av_freep(&p->avctx->slice_offset);

        }



        av_freep(&p->avctx->internal);

        av_freep(&p->avctx);

    }



    av_freep(&fctx->threads);

    pthread_mutex_destroy(&fctx->buffer_mutex);

    av_freep(&avctx->internal->thread_ctx);



    if (avctx->priv_data && avctx->codec && avctx->codec->priv_class)

        av_opt_free(avctx->priv_data);

    avctx->codec = NULL;

}
