static int trim_filter_frame(AVFilterLink *inlink, AVFrame *frame)

{

    AVFilterContext *ctx = inlink->dst;

    TrimContext       *s = ctx->priv;

    int drop;



    /* drop everything if EOF has already been returned */

    if (s->eof) {

        av_frame_free(&frame);

        return 0;

    }



    if (s->start_frame >= 0 || s->start_pts != AV_NOPTS_VALUE) {

        drop = 1;

        if (s->start_frame >= 0 && s->nb_frames >= s->start_frame)

            drop = 0;

        if (s->start_pts != AV_NOPTS_VALUE && frame->pts != AV_NOPTS_VALUE &&

            frame->pts >= s->start_pts)

            drop = 0;

        if (drop)

            goto drop;

    }



    if (s->first_pts == AV_NOPTS_VALUE && frame->pts != AV_NOPTS_VALUE)

        s->first_pts = frame->pts;



    if (s->end_frame != INT64_MAX || s->end_pts != AV_NOPTS_VALUE || s->duration_tb) {

        drop = 1;



        if (s->end_frame != INT64_MAX && s->nb_frames < s->end_frame)

            drop = 0;

        if (s->end_pts != AV_NOPTS_VALUE && frame->pts != AV_NOPTS_VALUE &&

            frame->pts < s->end_pts)

            drop = 0;

        if (s->duration_tb && frame->pts != AV_NOPTS_VALUE &&

            frame->pts - s->first_pts < s->duration_tb)

            drop = 0;



        if (drop) {

            s->eof = 1;

            goto drop;

        }

    }



    s->nb_frames++;

    s->got_output = 1;



    return ff_filter_frame(ctx->outputs[0], frame);



drop:

    s->nb_frames++;

    av_frame_free(&frame);

    return 0;

}
