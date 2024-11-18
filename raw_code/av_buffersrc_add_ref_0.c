int av_buffersrc_add_ref(AVFilterContext *buffer_filter,

                         AVFilterBufferRef *picref, int flags)

{

    BufferSourceContext *c = buffer_filter->priv;

    AVFilterLink *outlink = buffer_filter->outputs[0];

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

        /* TODO reindent */

    if (picref->video->w != c->w || picref->video->h != c->h || picref->format != c->pix_fmt) {

        AVFilterContext *scale = buffer_filter->outputs[0]->dst;

        AVFilterLink *link;

        char scale_param[1024];



        av_log(buffer_filter, AV_LOG_INFO,

               "Buffer video input changed from size:%dx%d fmt:%s to size:%dx%d fmt:%s\n",

               c->w, c->h, av_pix_fmt_descriptors[c->pix_fmt].name,

               picref->video->w, picref->video->h, av_pix_fmt_descriptors[picref->format].name);



        if (!scale || strcmp(scale->filter->name, "scale")) {

            AVFilter *f = avfilter_get_by_name("scale");



            av_log(buffer_filter, AV_LOG_INFO, "Inserting scaler filter\n");

            if ((ret = avfilter_open(&scale, f, "Input equalizer")) < 0)

                return ret;



            c->scale = scale;



            snprintf(scale_param, sizeof(scale_param)-1, "%d:%d:%s", c->w, c->h, c->sws_param);

            if ((ret = avfilter_init_filter(scale, scale_param, NULL)) < 0) {

                return ret;

            }



            if ((ret = avfilter_insert_filter(buffer_filter->outputs[0], scale, 0, 0)) < 0) {

                return ret;

            }

            scale->outputs[0]->time_base = scale->inputs[0]->time_base;



            scale->outputs[0]->format= c->pix_fmt;

        } else if (!strcmp(scale->filter->name, "scale")) {

            snprintf(scale_param, sizeof(scale_param)-1, "%d:%d:%s",

                     scale->outputs[0]->w, scale->outputs[0]->h, c->sws_param);

            scale->filter->init(scale, scale_param, NULL);

        }



        c->pix_fmt = scale->inputs[0]->format = picref->format;

        c->w       = scale->inputs[0]->w      = picref->video->w;

        c->h       = scale->inputs[0]->h      = picref->video->h;



        link = scale->outputs[0];

        if ((ret =  link->srcpad->config_props(link)) < 0)

            return ret;

    }

    }



    if (flags & AV_BUFFERSRC_FLAG_NO_COPY) {

        buf = picref;

    } else {

    buf = avfilter_get_video_buffer(outlink, AV_PERM_WRITE,

                                    picref->video->w, picref->video->h);

    av_image_copy(buf->data, buf->linesize,

                  (void*)picref->data, picref->linesize,

                  picref->format, picref->video->w, picref->video->h);

    avfilter_copy_buffer_ref_props(buf, picref);



    }



    if ((ret = av_fifo_generic_write(c->fifo, &buf, sizeof(buf), NULL)) < 0) {

        if (buf != picref)

            avfilter_unref_buffer(buf);

        return ret;

    }

    c->nb_failed_requests = 0;



    return 0;

}
