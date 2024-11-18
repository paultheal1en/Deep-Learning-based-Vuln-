static attribute_align_arg void *frame_worker_thread(void *arg)

{

    PerThreadContext *p = arg;

    FrameThreadContext *fctx = p->parent;

    AVCodecContext *avctx = p->avctx;

    AVCodec *codec = avctx->codec;



    while (1) {

        int i;

        if (p->state == STATE_INPUT_READY && !fctx->die) {

            pthread_mutex_lock(&p->mutex);

            while (p->state == STATE_INPUT_READY && !fctx->die)

                pthread_cond_wait(&p->input_cond, &p->mutex);

            pthread_mutex_unlock(&p->mutex);

        }



        if (fctx->die) break;



        if (!codec->update_thread_context && (avctx->thread_safe_callbacks || avctx->get_buffer == avcodec_default_get_buffer))

            ff_thread_finish_setup(avctx);



        pthread_mutex_lock(&p->mutex);

        avcodec_get_frame_defaults(&p->frame);

        p->got_frame = 0;

        p->result = codec->decode(avctx, &p->frame, &p->got_frame, &p->avpkt);



        if (p->state == STATE_SETTING_UP) ff_thread_finish_setup(avctx);



        pthread_mutex_lock(&p->progress_mutex);

        for (i = 0; i < MAX_BUFFERS; i++)

            if (p->progress_used[i]) {

                p->progress[i][0] = INT_MAX;

                p->progress[i][1] = INT_MAX;

            }

        p->state = STATE_INPUT_READY;



        pthread_cond_broadcast(&p->progress_cond);

        pthread_cond_signal(&p->output_cond);

        pthread_mutex_unlock(&p->progress_mutex);



        pthread_mutex_unlock(&p->mutex);

    }



    return NULL;

}
