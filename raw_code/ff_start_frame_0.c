int ff_start_frame(AVFilterLink *link, AVFilterBufferRef *picref)

{

    int (*start_frame)(AVFilterLink *, AVFilterBufferRef *);

    AVFilterPad *src = link->srcpad;

    AVFilterPad *dst = link->dstpad;

    int ret, perms;

    AVFilterCommand *cmd= link->dst->command_queue;

    int64_t pts;



    FF_TPRINTF_START(NULL, start_frame); ff_tlog_link(NULL, link, 0); ff_tlog(NULL, " "); ff_tlog_ref(NULL, picref, 1);



    av_assert1(picref->format                     == link->format);

    av_assert1(picref->video->w                   == link->w);

    av_assert1(picref->video->h                   == link->h);



    if (link->closed) {

        avfilter_unref_buffer(picref);

        return AVERROR_EOF;

    }



    if (!(start_frame = dst->start_frame))

        start_frame = default_start_frame;



    av_assert1((picref->perms & src->min_perms) == src->min_perms);

    picref->perms &= ~ src->rej_perms;

    perms = picref->perms;



    if (picref->linesize[0] < 0)

        perms |= AV_PERM_NEG_LINESIZES;

    /* prepare to copy the picture if it has insufficient permissions */

    if ((dst->min_perms & perms) != dst->min_perms || dst->rej_perms & perms) {

        av_log(link->dst, AV_LOG_DEBUG,

                "frame copy needed (have perms %x, need %x, reject %x)\n",

                picref->perms,

                link->dstpad->min_perms, link->dstpad->rej_perms);



        link->cur_buf = ff_get_video_buffer(link, dst->min_perms, link->w, link->h);

        if (!link->cur_buf) {

            avfilter_unref_bufferp(&picref);

            return AVERROR(ENOMEM);

        }



        link->src_buf = picref;

        avfilter_copy_buffer_ref_props(link->cur_buf, link->src_buf);



        /* copy palette if required */

        if (av_pix_fmt_descriptors[link->format].flags & PIX_FMT_PAL)

            memcpy(link->cur_buf->data[1], link->src_buf-> data[1], AVPALETTE_SIZE);

    }

    else

        link->cur_buf = picref;



    link->cur_buf_copy = link->cur_buf;



    while(cmd && cmd->time <= picref->pts * av_q2d(link->time_base)){

        av_log(link->dst, AV_LOG_DEBUG,

               "Processing command time:%f command:%s arg:%s\n",

               cmd->time, cmd->command, cmd->arg);

        avfilter_process_command(link->dst, cmd->command, cmd->arg, 0, 0, cmd->flags);

        ff_command_queue_pop(link->dst);

        cmd= link->dst->command_queue;

    }

    pts = link->cur_buf->pts;

    ret = start_frame(link, link->cur_buf);

    ff_update_link_current_pts(link, pts);

    if (ret < 0)

        clear_link(link);

    else

        /* incoming buffers must not be freed in start frame,

           because they can still be in use by the automatic copy mechanism */

        av_assert1(link->cur_buf_copy->buf->refcount > 0);



    return ret;

}
