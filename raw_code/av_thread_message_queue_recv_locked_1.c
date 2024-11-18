static int av_thread_message_queue_recv_locked(AVThreadMessageQueue *mq,

                                               void *msg,

                                               unsigned flags)

{

    while (!mq->err_recv && av_fifo_size(mq->fifo) < mq->elsize) {

        if ((flags & AV_THREAD_MESSAGE_NONBLOCK))

            return AVERROR(EAGAIN);

        pthread_cond_wait(&mq->cond, &mq->lock);

    }

    if (av_fifo_size(mq->fifo) < mq->elsize)

        return mq->err_recv;

    av_fifo_generic_read(mq->fifo, msg, mq->elsize, NULL);

    pthread_cond_signal(&mq->cond);

    return 0;

}
