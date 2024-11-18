static void select_frame(AVFilterContext *ctx, AVFrame *frame)

{

    SelectContext *select = ctx->priv;

    AVFilterLink *inlink = ctx->inputs[0];

    double res;



    if (isnan(select->var_values[VAR_START_PTS]))

        select->var_values[VAR_START_PTS] = TS2D(frame->pts);

    if (isnan(select->var_values[VAR_START_T]))

        select->var_values[VAR_START_T] = TS2D(frame->pts) * av_q2d(inlink->time_base);



    select->var_values[VAR_N  ] = inlink->frame_count;

    select->var_values[VAR_PTS] = TS2D(frame->pts);

    select->var_values[VAR_T  ] = TS2D(frame->pts) * av_q2d(inlink->time_base);

    select->var_values[VAR_POS] = av_frame_get_pkt_pos(frame) == -1 ? NAN : av_frame_get_pkt_pos(frame);



    switch (inlink->type) {

    case AVMEDIA_TYPE_AUDIO:

        select->var_values[VAR_SAMPLES_N] = frame->nb_samples;

        break;



    case AVMEDIA_TYPE_VIDEO:

        select->var_values[VAR_INTERLACE_TYPE] =

            !frame->interlaced_frame ? INTERLACE_TYPE_P :

        frame->top_field_first ? INTERLACE_TYPE_T : INTERLACE_TYPE_B;

        select->var_values[VAR_PICT_TYPE] = frame->pict_type;

#if CONFIG_AVCODEC

        if (select->do_scene_detect) {

            char buf[32];

            select->var_values[VAR_SCENE] = get_scene_score(ctx, frame);

            // TODO: document metadata

            snprintf(buf, sizeof(buf), "%f", select->var_values[VAR_SCENE]);

            av_dict_set(avpriv_frame_get_metadatap(frame), "lavfi.scene_score", buf, 0);

        }

#endif

        break;

    }



    select->select = res = av_expr_eval(select->expr, select->var_values, NULL);

    av_log(inlink->dst, AV_LOG_DEBUG,

           "n:%f pts:%f t:%f key:%d",

           select->var_values[VAR_N],

           select->var_values[VAR_PTS],

           select->var_values[VAR_T],

           (int)select->var_values[VAR_KEY]);



    switch (inlink->type) {

    case AVMEDIA_TYPE_VIDEO:

        av_log(inlink->dst, AV_LOG_DEBUG, " interlace_type:%c pict_type:%c scene:%f",

               select->var_values[VAR_INTERLACE_TYPE] == INTERLACE_TYPE_P ? 'P' :

               select->var_values[VAR_INTERLACE_TYPE] == INTERLACE_TYPE_T ? 'T' :

               select->var_values[VAR_INTERLACE_TYPE] == INTERLACE_TYPE_B ? 'B' : '?',

               av_get_picture_type_char(select->var_values[VAR_PICT_TYPE]),

               select->var_values[VAR_SCENE]);

        break;

    case AVMEDIA_TYPE_AUDIO:

        av_log(inlink->dst, AV_LOG_DEBUG, " samples_n:%d consumed_samples_n:%d",

               (int)select->var_values[VAR_SAMPLES_N],

               (int)select->var_values[VAR_CONSUMED_SAMPLES_N]);

        break;

    }



    if (res == 0) {

        select->select_out = -1; /* drop */

    } else if (isnan(res) || res < 0) {

        select->select_out = 0; /* first output */

    } else {

        select->select_out = FFMIN(ceilf(res)-1, select->nb_outputs-1); /* other outputs */

    }



    av_log(inlink->dst, AV_LOG_DEBUG, " -> select:%f select_out:%d\n", res, select->select_out);



    if (res) {

        select->var_values[VAR_PREV_SELECTED_N]   = select->var_values[VAR_N];

        select->var_values[VAR_PREV_SELECTED_PTS] = select->var_values[VAR_PTS];

        select->var_values[VAR_PREV_SELECTED_T]   = select->var_values[VAR_T];

        select->var_values[VAR_SELECTED_N] += 1.0;

        if (inlink->type == AVMEDIA_TYPE_AUDIO)

            select->var_values[VAR_CONSUMED_SAMPLES_N] += frame->nb_samples;

    }



    select->var_values[VAR_PREV_PTS] = select->var_values[VAR_PTS];

    select->var_values[VAR_PREV_T]   = select->var_values[VAR_T];

}
