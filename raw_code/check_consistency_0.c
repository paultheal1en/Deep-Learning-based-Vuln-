static void check_consistency(FFFrameQueue *fq)

{

#if ASSERT_LEVEL >= 2

    uint64_t nb_samples = 0;

    size_t i;



    av_assert0(fq->queued == fq->total_frames_head - fq->total_frames_tail);

    for (i = 0; i < fq->queued; i++)

        nb_samples += bucket(fq, i)->frame->nb_samples;

    av_assert0(nb_samples == fq->total_samples_head - fq->total_samples_tail);

#endif

}
