void av_thread_message_flush(AVThreadMessageQueue *mq)

{

#if HAVE_THREADS

    int used, off;

    void *free_func = mq->free_func;



    pthread_mutex_lock(&mq->lock);

    used = av_fifo_size(mq->fifo);

    if (free_func)

        for (off = 0; off < used; off += mq->elsize)

            av_fifo_generic_peek_at(mq->fifo, mq, off, mq->elsize, free_func_wrap);

    av_fifo_drain(mq->fifo, used);

    pthread_cond_broadcast(&mq->cond);

    pthread_mutex_unlock(&mq->lock);

#endif /* HAVE_THREADS */

}
