void ff_af_queue_remove(AudioFrameQueue *afq, int nb_samples, int64_t *pts,

                        int *duration)

{

    int64_t out_pts = AV_NOPTS_VALUE;

    int removed_samples = 0;



#ifdef DEBUG

    ff_af_queue_log_state(afq);

#endif



    /* get output pts from the next frame or generated pts */

    if (afq->frame_queue) {

        if (afq->frame_queue->pts != AV_NOPTS_VALUE)

            out_pts = afq->frame_queue->pts - afq->remaining_delay;

    } else {

        if (afq->next_pts != AV_NOPTS_VALUE)

            out_pts = afq->next_pts - afq->remaining_delay;

    }

    if (pts) {

        if (out_pts != AV_NOPTS_VALUE)

            *pts = ff_samples_to_time_base(afq->avctx, out_pts);

        else

            *pts = AV_NOPTS_VALUE;

    }



    /* if the delay is larger than the packet duration, we use up delay samples

       for the output packet and leave all frames in the queue */

    if (afq->remaining_delay >= nb_samples) {

        removed_samples      += nb_samples;

        afq->remaining_delay -= nb_samples;

    }

    /* remove frames from the queue until we have enough to cover the

       requested number of samples or until the queue is empty */

    while (removed_samples < nb_samples && afq->frame_queue) {

        removed_samples += afq->frame_queue->duration;

        delete_next_frame(afq);

    }

    afq->remaining_samples -= removed_samples;



    /* if there are no frames left and we have room for more samples, use

       any remaining delay samples */

    if (removed_samples < nb_samples && afq->remaining_samples > 0) {

        int add_samples = FFMIN(afq->remaining_samples,

                                nb_samples - removed_samples);

        removed_samples        += add_samples;

        afq->remaining_samples -= add_samples;

    }

    if (removed_samples > nb_samples)

        av_log(afq->avctx, AV_LOG_WARNING, "frame_size is too large\n");

    if (duration)

        *duration = ff_samples_to_time_base(afq->avctx, removed_samples);

}
