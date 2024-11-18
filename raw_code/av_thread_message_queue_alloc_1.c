int av_thread_message_queue_alloc(AVThreadMessageQueue **mq,

                                  unsigned nelem,

                                  unsigned elsize)

{

#if HAVE_THREADS

    AVThreadMessageQueue *rmq;

    int ret = 0;



    if (nelem > INT_MAX / elsize)

        return AVERROR(EINVAL);

    if (!(rmq = av_mallocz(sizeof(*rmq))))

        return AVERROR(ENOMEM);

    if ((ret = pthread_mutex_init(&rmq->lock, NULL))) {

        av_free(rmq);

        return AVERROR(ret);

    }

    if ((ret = pthread_cond_init(&rmq->cond, NULL))) {

        pthread_mutex_destroy(&rmq->lock);

        av_free(rmq);

        return AVERROR(ret);

    }

    if (!(rmq->fifo = av_fifo_alloc(elsize * nelem))) {

        pthread_cond_destroy(&rmq->cond);

        pthread_mutex_destroy(&rmq->lock);

        av_free(rmq);

        return AVERROR(ret);

    }

    rmq->elsize = elsize;

    *mq = rmq;

    return 0;

#else

    *mq = NULL;

    return AVERROR(ENOSYS);

#endif /* HAVE_THREADS */

}
