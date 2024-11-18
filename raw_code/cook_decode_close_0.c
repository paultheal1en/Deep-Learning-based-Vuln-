static int cook_decode_close(AVCodecContext *avctx)

{

    int i;

    COOKContext *q = avctx->priv_data;

    av_log(NULL,AV_LOG_DEBUG, "Deallocating memory.\n");



    /* Free allocated memory buffers. */

    av_free(q->mlt_window);

    av_free(q->mlt_precos);

    av_free(q->mlt_presin);

    av_free(q->mlt_postcos);

    av_free(q->frame_reorder_index);

    av_free(q->frame_reorder_buffer);

    av_free(q->decoded_bytes_buffer);



    /* Free the transform. */

    ff_fft_end(&q->fft_ctx);



    /* Free the VLC tables. */

    for (i=0 ; i<13 ; i++) {

        free_vlc(&q->envelope_quant_index[i]);

    }

    for (i=0 ; i<7 ; i++) {

        free_vlc(&q->sqvh[i]);

    }

    if(q->nb_channels==2 && q->joint_stereo==1 ){

        free_vlc(&q->ccpl);

    }



    av_log(NULL,AV_LOG_DEBUG,"Memory deallocated.\n");



    return 0;

}
