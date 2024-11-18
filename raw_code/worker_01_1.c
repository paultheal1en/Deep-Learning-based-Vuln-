static void * attribute_align_arg worker(void *v){

    AVCodecContext *avctx = v;

    ThreadContext *c = avctx->internal->frame_thread_encoder;

    AVPacket *pkt = NULL;



    while(!c->exit){

        int got_packet, ret;

        AVFrame *frame;

        Task task;



        if(!pkt) pkt= av_mallocz(sizeof(*pkt));

        if(!pkt) continue;

        av_init_packet(pkt);



        pthread_mutex_lock(&c->task_fifo_mutex);

        while (av_fifo_size(c->task_fifo) <= 0 || c->exit) {

            if(c->exit){

                pthread_mutex_unlock(&c->task_fifo_mutex);

                goto end;

            }

            pthread_cond_wait(&c->task_fifo_cond, &c->task_fifo_mutex);

        }

        av_fifo_generic_read(c->task_fifo, &task, sizeof(task), NULL);

        pthread_mutex_unlock(&c->task_fifo_mutex);

        frame = task.indata;



        ret = avcodec_encode_video2(avctx, pkt, frame, &got_packet);

        pthread_mutex_lock(&c->buffer_mutex);

        av_frame_unref(frame);

        pthread_mutex_unlock(&c->buffer_mutex);

        av_frame_free(&frame);

        if(got_packet) {

            int ret2 = av_dup_packet(pkt);

            if (ret >= 0 && ret2 < 0)

                ret = ret2;

        } else {

            pkt->data = NULL;

            pkt->size = 0;

        }

        pthread_mutex_lock(&c->finished_task_mutex);

        c->finished_tasks[task.index].outdata = pkt; pkt = NULL;

        c->finished_tasks[task.index].return_code = ret;

        pthread_cond_signal(&c->finished_task_cond);

        pthread_mutex_unlock(&c->finished_task_mutex);

    }

end:

    av_free(pkt);

    pthread_mutex_lock(&c->buffer_mutex);

    avcodec_close(avctx);

    pthread_mutex_unlock(&c->buffer_mutex);

    av_freep(&avctx);

    return NULL;

}
