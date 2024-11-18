int av_buffersrc_add_ref(AVFilterContext *buffer_filter,

                         AVFilterBufferRef *picref, int flags)

{

    BufferSourceContext *c = buffer_filter->priv;

    AVFilterBufferRef *buf;

    int ret;



    if (!picref) {

        c->eof = 1;

        return 0;

    } else if (c->eof)

        return AVERROR(EINVAL);



    if (!av_fifo_space(c->fifo) &&

        (ret = av_fifo_realloc2(c->fifo, av_fifo_size(c->fifo) +

                                         sizeof(buf))) < 0)

        return ret;



    if (!(flags & AV_BUFFERSRC_FLAG_NO_CHECK_FORMAT)) {

        ret = check_format_change(buffer_filter, picref);

        if (ret < 0)

            return ret;

    }

    if (flags & AV_BUFFERSRC_FLAG_NO_COPY)

        buf = picref;

    else

        buf = copy_buffer_ref(buffer_filter, picref);





    if ((ret = av_fifo_generic_write(c->fifo, &buf, sizeof(buf), NULL)) < 0) {

        if (buf != picref)

            avfilter_unref_buffer(buf);

        return ret;

    }

    c->nb_failed_requests = 0;



    return 0;

}