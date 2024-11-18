void ff_af_queue_log_state(AudioFrameQueue *afq)

{

    AudioFrame *f;

    av_log(afq->avctx, AV_LOG_DEBUG, "remaining delay   = %d\n",

           afq->remaining_delay);

    av_log(afq->avctx, AV_LOG_DEBUG, "remaining samples = %d\n",

           afq->remaining_samples);

    av_log(afq->avctx, AV_LOG_DEBUG, "frames:\n");

    f = afq->frame_queue;

    while (f) {

        av_log(afq->avctx, AV_LOG_DEBUG, "  [ pts=%9"PRId64" duration=%d ]\n",

               f->pts, f->duration);

        f = f->next;

    }

}
