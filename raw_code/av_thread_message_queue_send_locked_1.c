static int av_thread_message_queue_send_locked(AVThreadMessageQueue *mq,

                                               void *msg,

                                               unsigned flags)

{

    while (!mq->err_send && av_fifo_space(mq->fifo) < mq->elsize) {

        if ((flags & AV_THREAD_MESSAGE_NONBLOCK))

            return AVERROR(EAGAIN);

        pthread_cond_wait(&mq->cond, &mq->lock);

    }

    if (mq->err_send)

        return mq->err_send;

    av_fifo_generic_write(mq->fifo, msg, mq->elsize, NULL);

    pthread_cond_signal(&mq->cond);

    return 0;

}
