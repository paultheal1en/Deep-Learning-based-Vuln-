int ff_v4l2_m2m_codec_full_reinit(V4L2m2mContext *s)

{

    void *log_ctx = s->avctx;

    int ret;



    av_log(log_ctx, AV_LOG_DEBUG, "%s full reinit\n", s->devname);



    /* wait for pending buffer references */

    if (atomic_load(&s->refcount))

        while(sem_wait(&s->refsync) == -1 && errno == EINTR);



    /* close the driver */

    ff_v4l2_m2m_codec_end(s->avctx);



    /* start again now that we know the stream dimensions */

    s->draining = 0;

    s->reinit = 0;



    s->fd = open(s->devname, O_RDWR | O_NONBLOCK, 0);

    if (s->fd < 0)

        return AVERROR(errno);



    ret = v4l2_prepare_contexts(s);

    if (ret < 0)

        goto error;



    /* if a full re-init was requested - probe didn't run - we need to populate

     * the format for each context

     */

    ret = ff_v4l2_context_get_format(&s->output);

    if (ret) {

        av_log(log_ctx, AV_LOG_DEBUG, "v4l2 output format not supported\n");

        goto error;

    }



    ret = ff_v4l2_context_get_format(&s->capture);

    if (ret) {

        av_log(log_ctx, AV_LOG_DEBUG, "v4l2 capture format not supported\n");

        goto error;

    }



    ret = ff_v4l2_context_set_format(&s->output);

    if (ret) {

        av_log(log_ctx, AV_LOG_ERROR, "can't set v4l2 output format\n");

        goto error;

    }



    ret = ff_v4l2_context_set_format(&s->capture);

    if (ret) {

        av_log(log_ctx, AV_LOG_ERROR, "can't to set v4l2 capture format\n");

        goto error;

    }



    ret = ff_v4l2_context_init(&s->output);

    if (ret) {

        av_log(log_ctx, AV_LOG_ERROR, "no v4l2 output context's buffers\n");

        goto error;

    }



    /* decoder's buffers need to be updated at a later stage */

    if (!av_codec_is_decoder(s->avctx->codec)) {

        ret = ff_v4l2_context_init(&s->capture);

        if (ret) {

            av_log(log_ctx, AV_LOG_ERROR, "no v4l2 capture context's buffers\n");

            goto error;

        }

    }



    return 0;



error:

    if (close(s->fd) < 0) {

        ret = AVERROR(errno);

        av_log(log_ctx, AV_LOG_ERROR, "error closing %s (%s)\n",

            s->devname, av_err2str(AVERROR(errno)));

    }

    s->fd = -1;



    return ret;

}
