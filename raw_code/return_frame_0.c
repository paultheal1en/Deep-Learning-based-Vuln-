static void return_frame(AVFilterContext *ctx, int is_second)

{

    YADIFContext *yadif = ctx->priv;

    AVFilterLink *link= ctx->outputs[0];

    int tff;



    if (yadif->parity == -1) {

        tff = yadif->cur->video->interlaced ?

            yadif->cur->video->top_field_first : 1;

    } else {

        tff = yadif->parity^1;

    }



    if (is_second) {

        yadif->out = ff_get_video_buffer(link, AV_PERM_WRITE | AV_PERM_PRESERVE |

                                         AV_PERM_REUSE, link->w, link->h);

        avfilter_copy_buffer_ref_props(yadif->out, yadif->cur);

        yadif->out->video->interlaced = 0;

    }



    if (!yadif->csp)

        yadif->csp = &av_pix_fmt_descriptors[link->format];

    if (yadif->csp->comp[0].depth_minus1 / 8 == 1)

        yadif->filter_line = filter_line_c_16bit;



    filter(ctx, yadif->out, tff ^ !is_second, tff);



    if (is_second) {

        int64_t cur_pts  = yadif->cur->pts;

        int64_t next_pts = yadif->next->pts;



        if (next_pts != AV_NOPTS_VALUE && cur_pts != AV_NOPTS_VALUE) {

            yadif->out->pts = cur_pts + next_pts;

        } else {

            yadif->out->pts = AV_NOPTS_VALUE;

        }

        ff_start_frame(ctx->outputs[0], yadif->out);

    }

    ff_draw_slice(ctx->outputs[0], 0, link->h, 1);

    ff_end_frame(ctx->outputs[0]);



    yadif->frame_pending = (yadif->mode&1) && !is_second;

}
