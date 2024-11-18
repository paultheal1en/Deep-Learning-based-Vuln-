AVFrame *ff_framequeue_take(FFFrameQueue *fq)

{

    FFFrameBucket *b;



    check_consistency(fq);

    av_assert1(fq->queued);

    b = bucket(fq, 0);

    fq->queued--;

    fq->tail++;

    fq->tail &= fq->allocated - 1;

    fq->total_frames_tail++;

    fq->total_samples_tail += b->frame->nb_samples;


    check_consistency(fq);

    return b->frame;

}