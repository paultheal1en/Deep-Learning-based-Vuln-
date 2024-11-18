int avcodec_close(AVCodecContext *avctx)

{

    /* If there is a user-supplied mutex locking routine, call it. */

    if (ff_lockmgr_cb) {

        if ((*ff_lockmgr_cb)(&codec_mutex, AV_LOCK_OBTAIN))

            return -1;

    }



    entangled_thread_counter++;

    if(entangled_thread_counter != 1){

        av_log(avctx, AV_LOG_ERROR, "insufficient thread locking around avcodec_open/close()\n");

        entangled_thread_counter--;

        return -1;

    }



    if (HAVE_THREADS && avctx->thread_opaque)

        avcodec_thread_free(avctx);

    if (avctx->codec->close)

        avctx->codec->close(avctx);

    avcodec_default_free_buffers(avctx);

    av_freep(&avctx->priv_data);

    avctx->codec = NULL;

    entangled_thread_counter--;



    /* Release any user-supplied mutex. */

    if (ff_lockmgr_cb) {

        (*ff_lockmgr_cb)(&codec_mutex, AV_LOCK_RELEASE);

    }

    return 0;

}
