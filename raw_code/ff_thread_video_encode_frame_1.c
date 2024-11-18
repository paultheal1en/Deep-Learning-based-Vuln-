int ff_thread_video_encode_frame(AVCodecContext *avctx, AVPacket *pkt, const AVFrame *frame, int *got_packet_ptr){

    ThreadContext *c = avctx->internal->frame_thread_encoder;

    Task task;

    int ret;



    av_assert1(!*got_packet_ptr);



    if(frame){

        if(!(avctx->flags & CODEC_FLAG_INPUT_PRESERVED)){

            AVFrame *new = avcodec_alloc_frame();

            if(!new)

                return AVERROR(ENOMEM);

            pthread_mutex_lock(&c->buffer_mutex);

            ret = c->parent_avctx->get_buffer(c->parent_avctx, new);

            pthread_mutex_unlock(&c->buffer_mutex);

            if(ret<0)

                return ret;

            new->pts = frame->pts;

            new->quality = frame->quality;

            new->pict_type = frame->pict_type;

            av_image_copy(new->data, new->linesize, (const uint8_t **)frame->data, frame->linesize,

                          avctx->pix_fmt, avctx->width, avctx->height);

            frame = new;

        }



        task.index = c->task_index;

        task.indata = (void*)frame;

        pthread_mutex_lock(&c->task_fifo_mutex);

        av_fifo_generic_write(c->task_fifo, &task, sizeof(task), NULL);

        pthread_cond_signal(&c->task_fifo_cond);

        pthread_mutex_unlock(&c->task_fifo_mutex);



        c->task_index = (c->task_index+1) % BUFFER_SIZE;



        if(!c->finished_tasks[c->finished_task_index].outdata && (c->task_index - c->finished_task_index) % BUFFER_SIZE <= avctx->thread_count)

            return 0;

    }



    if(c->task_index == c->finished_task_index)

        return 0;



    pthread_mutex_lock(&c->finished_task_mutex);

    while (!c->finished_tasks[c->finished_task_index].outdata) {

        pthread_cond_wait(&c->finished_task_cond, &c->finished_task_mutex);

    }

    task = c->finished_tasks[c->finished_task_index];

    *pkt = *(AVPacket*)(task.outdata);

    av_freep(&c->finished_tasks[c->finished_task_index].outdata);

    c->finished_task_index = (c->finished_task_index+1) % BUFFER_SIZE;

    pthread_mutex_unlock(&c->finished_task_mutex);



    *got_packet_ptr = 1;



    return task.return_code;

}
