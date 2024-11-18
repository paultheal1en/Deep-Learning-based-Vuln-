static double get_scene_score(AVFilterContext *ctx, AVFrame *crnt, AVFrame *next)

{

    FrameRateContext *s = ctx->priv;

    double ret = 0;



    ff_dlog(ctx, "get_scene_score()\n");



    if (crnt->height == next->height &&

        crnt->width  == next->width) {

        int64_t sad;

        double mafd, diff;



        ff_dlog(ctx, "get_scene_score() process\n");

        if (s->bitdepth == 8)

            sad = scene_sad8(s, crnt->data[0], crnt->linesize[0], next->data[0], next->linesize[0], crnt->height);

        else

            sad = scene_sad16(s, (const uint16_t*)crnt->data[0], crnt->linesize[0] >> 1, (const uint16_t*)next->data[0], next->linesize[0] >> 1, crnt->height);



        mafd = (double)sad * 100.0 / (crnt->height * crnt->width) / (1 << s->bitdepth);

        diff = fabs(mafd - s->prev_mafd);

        ret  = av_clipf(FFMIN(mafd, diff), 0, 100.0);

        s->prev_mafd = mafd;

    }

    ff_dlog(ctx, "get_scene_score() result is:%f\n", ret);

    return ret;

}
