static void start_frame(AVFilterLink *link, AVFilterBufferRef *picref)

{

    AVFilterContext *ctx = link->dst;

    CropContext *crop = ctx->priv;

    AVFilterBufferRef *ref2;

    int i;



    picref->video->w = crop->w;

    picref->video->h = crop->h;



    ref2 = avfilter_ref_buffer(picref, ~0);



    crop->var_values[VAR_T] = picref->pts == AV_NOPTS_VALUE ?

        NAN : picref->pts * av_q2d(link->time_base);

    crop->var_values[VAR_POS] = picref->pos == -1 ? NAN : picref->pos;

    crop->var_values[VAR_X] = av_expr_eval(crop->x_pexpr, crop->var_values, NULL);

    crop->var_values[VAR_Y] = av_expr_eval(crop->y_pexpr, crop->var_values, NULL);

    crop->var_values[VAR_X] = av_expr_eval(crop->x_pexpr, crop->var_values, NULL);



    normalize_double(&crop->x, crop->var_values[VAR_X]);

    normalize_double(&crop->y, crop->var_values[VAR_Y]);



    if (crop->x < 0) crop->x = 0;

    if (crop->y < 0) crop->y = 0;

    if ((unsigned)crop->x + (unsigned)crop->w > link->w) crop->x = link->w - crop->w;

    if ((unsigned)crop->y + (unsigned)crop->h > link->h) crop->y = link->h - crop->h;

    crop->x &= ~((1 << crop->hsub) - 1);

    crop->y &= ~((1 << crop->vsub) - 1);




    av_log(ctx, AV_LOG_DEBUG,

           "n:%d t:%f x:%d y:%d x+w:%d y+h:%d\n",

           (int)crop->var_values[VAR_N], crop->var_values[VAR_T], crop->x, crop->y, crop->x+crop->w, crop->y+crop->h);




    ref2->data[0] += crop->y * ref2->linesize[0];

    ref2->data[0] += crop->x * crop->max_step[0];



    if (!(av_pix_fmt_descriptors[link->format].flags & PIX_FMT_PAL)) {

        for (i = 1; i < 3; i ++) {

            if (ref2->data[i]) {

                ref2->data[i] += (crop->y >> crop->vsub) * ref2->linesize[i];

                ref2->data[i] += (crop->x * crop->max_step[i]) >> crop->hsub;

            }

        }

    }



    /* alpha plane */

    if (ref2->data[3]) {

        ref2->data[3] += crop->y * ref2->linesize[3];

        ref2->data[3] += crop->x * crop->max_step[3];

    }



    avfilter_start_frame(link->dst->outputs[0], ref2);

}