static int end_frame(AVFilterLink *inlink)

{

    AVFilterContext    *ctx = inlink->dst;

    FPSContext           *s = ctx->priv;

    AVFilterLink   *outlink = ctx->outputs[0];

    AVFilterBufferRef  *buf = inlink->cur_buf;

    int64_t delta;

    int i, ret;



    inlink->cur_buf = NULL;

    s->frames_in++;

    /* discard frames until we get the first timestamp */

    if (s->pts == AV_NOPTS_VALUE) {

        if (buf->pts != AV_NOPTS_VALUE) {

            write_to_fifo(s->fifo, buf);

            s->first_pts = s->pts = buf->pts;

        } else {

            av_log(ctx, AV_LOG_WARNING, "Discarding initial frame(s) with no "

                   "timestamp.\n");

            avfilter_unref_buffer(buf);

            s->drop++;

        }

        return 0;

    }



    /* now wait for the next timestamp */

    if (buf->pts == AV_NOPTS_VALUE) {

        return write_to_fifo(s->fifo, buf);

    }



    /* number of output frames */

    delta = av_rescale_q(buf->pts - s->pts, inlink->time_base,

                         outlink->time_base);



    if (delta < 1) {

        /* drop the frame and everything buffered except the first */

        AVFilterBufferRef *tmp;

        int drop = av_fifo_size(s->fifo)/sizeof(AVFilterBufferRef*);



        av_log(ctx, AV_LOG_DEBUG, "Dropping %d frame(s).\n", drop);

        s->drop += drop;



        av_fifo_generic_read(s->fifo, &tmp, sizeof(tmp), NULL);

        flush_fifo(s->fifo);

        ret = write_to_fifo(s->fifo, tmp);



        avfilter_unref_buffer(buf);

        return ret;

    }



    /* can output >= 1 frames */

    for (i = 0; i < delta; i++) {

        AVFilterBufferRef *buf_out;

        av_fifo_generic_read(s->fifo, &buf_out, sizeof(buf_out), NULL);



        /* duplicate the frame if needed */

        if (!av_fifo_size(s->fifo) && i < delta - 1) {

            av_log(ctx, AV_LOG_DEBUG, "Duplicating frame.\n");

            write_to_fifo(s->fifo, avfilter_ref_buffer(buf_out, AV_PERM_READ));

            s->dup++;

        }



        buf_out->pts = av_rescale_q(s->first_pts, inlink->time_base,

                                    outlink->time_base) + s->frames_out;



        if ((ret = ff_start_frame(outlink, buf_out)) < 0 ||

            (ret = ff_draw_slice(outlink, 0, outlink->h, 1)) < 0 ||

            (ret = ff_end_frame(outlink)) < 0) {

            avfilter_unref_bufferp(&buf);

            return ret;

        }



        s->frames_out++;

    }

    flush_fifo(s->fifo);



    ret = write_to_fifo(s->fifo, buf);

    s->pts = s->first_pts + av_rescale_q(s->frames_out, outlink->time_base, inlink->time_base);



    return ret;

}
