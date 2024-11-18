static int start_frame(AVFilterLink *link, AVFilterBufferRef *picref)

{

    AVFilterContext *ctx = link->dst;

    YADIFContext *yadif = ctx->priv;



    if (yadif->frame_pending)

        return_frame(ctx, 1);



    if (yadif->prev)

        avfilter_unref_buffer(yadif->prev);

    yadif->prev = yadif->cur;

    yadif->cur  = yadif->next;

    yadif->next = picref;



    if (!yadif->cur)

        return 0;



    if (yadif->auto_enable && !yadif->cur->video->interlaced) {

        yadif->out  = avfilter_ref_buffer(yadif->cur, AV_PERM_READ);

        avfilter_unref_bufferp(&yadif->prev);

        if (yadif->out->pts != AV_NOPTS_VALUE)

            yadif->out->pts *= 2;

        return ff_start_frame(ctx->outputs[0], yadif->out);

    }



    if (!yadif->prev)

        yadif->prev = avfilter_ref_buffer(yadif->cur, AV_PERM_READ);



    yadif->out = ff_get_video_buffer(ctx->outputs[0], AV_PERM_WRITE | AV_PERM_PRESERVE |

                                     AV_PERM_REUSE, link->w, link->h);



    avfilter_copy_buffer_ref_props(yadif->out, yadif->cur);

    yadif->out->video->interlaced = 0;

    if (yadif->out->pts != AV_NOPTS_VALUE)

        yadif->out->pts *= 2;

    return ff_start_frame(ctx->outputs[0], yadif->out);

}
