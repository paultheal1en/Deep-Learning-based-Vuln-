static void delete_next_frame(AudioFrameQueue *afq)

{

    AudioFrame *f = afq->frame_queue;

    if (f) {

        afq->frame_queue = f->next;

        f->next = NULL;

        av_freep(&f);

    }

}
