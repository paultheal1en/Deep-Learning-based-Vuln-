static attribute_align_arg void *frame_worker_thread(void *arg)

{

    PerThreadContext *p = arg;

    FrameThreadContext *fctx = p->parent;

    AVCodecContext *avctx = p->avctx;

    const AVCodec *codec = avctx->codec;



    pthread_mutex_lock(&p->mutex);

    while (1) {

            while (p->state == STATE_INPUT_READY && !fctx->die)

                pthread_cond_wait(&p->input_cond, &p->mutex);



        if (fctx->die) break;



        if (!codec->update_thread_context && (avctx->thread_safe_callbacks || (

#if FF_API_GET_BUFFER

            !avctx->get_buffer &&

#endif

            avctx->get_buffer2 == avcodec_default_get_buffer2)))

            ff_thread_finish_setup(avctx);



        avcodec_get_frame_defaults(&p->frame);

        p->got_frame = 0;

        p->result = codec->decode(avctx, &p->frame, &p->got_frame, &p->avpkt);



        /* many decoders assign whole AVFrames, thus overwriting extended_data;

         * make sure it's set correctly */

        p->frame.extended_data = p->frame.data;



        if (p->state == STATE_SETTING_UP) ff_thread_finish_setup(avctx);



        pthread_mutex_lock(&p->progress_mutex);

#if 0 //BUFREF-FIXME

        for (i = 0; i < MAX_BUFFERS; i++)

            if (p->progress_used[i] && (p->got_frame || p->result<0 || avctx->codec_id != AV_CODEC_ID_H264)) {

                p->progress[i][0] = INT_MAX;

                p->progress[i][1] = INT_MAX;

            }

#endif

        p->state = STATE_INPUT_READY;



        pthread_cond_broadcast(&p->progress_cond);

        pthread_cond_signal(&p->output_cond);

        pthread_mutex_unlock(&p->progress_mutex);

    }

    pthread_mutex_unlock(&p->mutex);



    return NULL;

}
