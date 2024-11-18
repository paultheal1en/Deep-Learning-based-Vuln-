int ff_frame_thread_encoder_init(AVCodecContext *avctx, AVDictionary *options){

    int i=0;

    ThreadContext *c;





    if(   !(avctx->thread_type & FF_THREAD_FRAME)

       || !(avctx->codec->capabilities & CODEC_CAP_INTRA_ONLY))

        return 0;



    if(   !avctx->thread_count

       && avctx->codec_id == AV_CODEC_ID_MJPEG

       && !(avctx->flags & CODEC_FLAG_QSCALE)) {

        av_log(avctx, AV_LOG_DEBUG,

               "Forcing thread count to 1 for MJPEG encoding, use -thread_type slice "

               "or a constant quantizer if you want to use multiple cpu cores\n");

        avctx->thread_count = 1;

    }

    if(   avctx->thread_count > 1

       && avctx->codec_id == AV_CODEC_ID_MJPEG

       && !(avctx->flags & CODEC_FLAG_QSCALE))

        av_log(avctx, AV_LOG_WARNING,

               "MJPEG CBR encoding works badly with frame multi-threading, consider "

               "using -threads 1, -thread_type slice or a constant quantizer.\n");

    if(!avctx->thread_count) {

        avctx->thread_count = av_cpu_count();

        avctx->thread_count = FFMIN(avctx->thread_count, MAX_THREADS);

    }



    if(avctx->thread_count <= 1)

        return 0;



    if (avctx->codec_id == AV_CODEC_ID_HUFFYUV ||

        avctx->codec_id == AV_CODEC_ID_FFVHUFF) {

        // huffyuv doesnt support these with multiple frame threads currently

        if (avctx->context_model > 0 || (avctx->flags & CODEC_FLAG_PASS1))

            return 0;

    }



    if(avctx->thread_count > MAX_THREADS)

        return AVERROR(EINVAL);



    av_assert0(!avctx->internal->frame_thread_encoder);

    c = avctx->internal->frame_thread_encoder = av_mallocz(sizeof(ThreadContext));

    if(!c)

        return AVERROR(ENOMEM);



    c->parent_avctx = avctx;



    c->task_fifo = av_fifo_alloc(sizeof(Task) * BUFFER_SIZE);

    if(!c->task_fifo)

        goto fail;



    pthread_mutex_init(&c->task_fifo_mutex, NULL);

    pthread_mutex_init(&c->finished_task_mutex, NULL);

    pthread_mutex_init(&c->buffer_mutex, NULL);

    pthread_cond_init(&c->task_fifo_cond, NULL);

    pthread_cond_init(&c->finished_task_cond, NULL);



    for(i=0; i<avctx->thread_count ; i++){

        AVDictionary *tmp = NULL;

        void *tmpv;

        AVCodecContext *thread_avctx = avcodec_alloc_context3(avctx->codec);

        if(!thread_avctx)

            goto fail;

        tmpv = thread_avctx->priv_data;

        *thread_avctx = *avctx;

        thread_avctx->priv_data = tmpv;

        thread_avctx->internal = NULL;

        memcpy(thread_avctx->priv_data, avctx->priv_data, avctx->codec->priv_data_size);

        thread_avctx->thread_count = 1;

        thread_avctx->active_thread_type &= ~FF_THREAD_FRAME;



        av_dict_copy(&tmp, options, 0);

        av_dict_set(&tmp, "threads", "1", 0);

        if(avcodec_open2(thread_avctx, avctx->codec, &tmp) < 0) {

            av_dict_free(&tmp);

            goto fail;

        }

        av_dict_free(&tmp);

        av_assert0(!thread_avctx->internal->frame_thread_encoder);

        thread_avctx->internal->frame_thread_encoder = c;

        if(pthread_create(&c->worker[i], NULL, worker, thread_avctx)) {

            goto fail;

        }

    }



    avctx->active_thread_type = FF_THREAD_FRAME;



    return 0;

fail:

    avctx->thread_count = i;

    av_log(avctx, AV_LOG_ERROR, "ff_frame_thread_encoder_init failed\n");

    ff_frame_thread_encoder_free(avctx);

    return -1;

}
