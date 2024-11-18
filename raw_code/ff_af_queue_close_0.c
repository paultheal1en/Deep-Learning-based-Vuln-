void ff_af_queue_close(AudioFrameQueue *afq)

{

    /* remove/free any remaining frames */

    while (afq->frame_queue)

        delete_next_frame(afq);

    memset(afq, 0, sizeof(*afq));

}
