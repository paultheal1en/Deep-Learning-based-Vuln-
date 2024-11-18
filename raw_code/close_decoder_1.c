static void close_decoder(QSVContext *q)

{

    QSVFrame *cur;



    if (q->session)

        MFXVideoDECODE_Close(q->session);



    while (q->async_fifo && av_fifo_size(q->async_fifo)) {

        QSVFrame *out_frame;

        mfxSyncPoint *sync;



        av_fifo_generic_read(q->async_fifo, &out_frame, sizeof(out_frame), NULL);

        av_fifo_generic_read(q->async_fifo, &sync,      sizeof(sync),      NULL);



        av_freep(&sync);

    }



    cur = q->work_frames;

    while (cur) {

        q->work_frames = cur->next;

        av_frame_free(&cur->frame);

        av_freep(&cur);

        cur = q->work_frames;

    }



    q->engine_ready   = 0;

    q->reinit_pending = 0;

}
