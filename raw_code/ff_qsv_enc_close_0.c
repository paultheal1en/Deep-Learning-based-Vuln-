int ff_qsv_enc_close(AVCodecContext *avctx, QSVEncContext *q)

{

    QSVFrame *cur;



    MFXVideoENCODE_Close(q->session);

    if (q->internal_session)

        MFXClose(q->internal_session);

    q->session          = NULL;

    q->internal_session = NULL;



    cur = q->work_frames;

    while (cur) {

        q->work_frames = cur->next;

        av_frame_free(&cur->frame);

        av_freep(&cur);

        cur = q->work_frames;

    }



    while (q->async_fifo && av_fifo_size(q->async_fifo)) {

        AVPacket pkt;

        mfxSyncPoint sync;

        mfxBitstream *bs;



        av_fifo_generic_read(q->async_fifo, &pkt,  sizeof(pkt),  NULL);

        av_fifo_generic_read(q->async_fifo, &sync, sizeof(sync), NULL);

        av_fifo_generic_read(q->async_fifo, &bs,   sizeof(bs),   NULL);



        av_freep(&bs);

        av_packet_unref(&pkt);

    }

    av_fifo_free(q->async_fifo);

    q->async_fifo = NULL;



    av_frame_free(&avctx->coded_frame);



    return 0;

}
