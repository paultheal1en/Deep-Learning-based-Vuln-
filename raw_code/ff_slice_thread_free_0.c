void ff_slice_thread_free(AVCodecContext *avctx)

{

    ThreadContext *c = avctx->thread_opaque;

    int i;



    pthread_mutex_lock(&c->current_job_lock);

    c->done = 1;

    pthread_cond_broadcast(&c->current_job_cond);

    pthread_mutex_unlock(&c->current_job_lock);



    for (i=0; i<avctx->thread_count; i++)

         pthread_join(c->workers[i], NULL);



    pthread_mutex_destroy(&c->current_job_lock);

    pthread_cond_destroy(&c->current_job_cond);

    pthread_cond_destroy(&c->last_job_cond);

    av_free(c->workers);

    av_freep(&avctx->thread_opaque);

}
