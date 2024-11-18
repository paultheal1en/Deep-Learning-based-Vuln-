static int init_filter_param(AVFilterContext *ctx, FilterParam *fp, const char *effect_type, int width)

{

    int z;

    const char *effect = fp->amount == 0 ? "none" : fp->amount < 0 ? "blur" : "sharpen";



    if  (!(fp->msize_x & fp->msize_y & 1)) {

        av_log(ctx, AV_LOG_ERROR,

               "Invalid even size for %s matrix size %dx%d\n",

               effect_type, fp->msize_x, fp->msize_y);

        return AVERROR(EINVAL);

    }



    av_log(ctx, AV_LOG_VERBOSE, "effect:%s type:%s msize_x:%d msize_y:%d amount:%0.2f\n",

           effect, effect_type, fp->msize_x, fp->msize_y, fp->amount / 65535.0);



    for (z = 0; z < 2 * fp->steps_y; z++)

        fp->sc[z] = av_malloc(sizeof(*(fp->sc[z])) * (width + 2 * fp->steps_x));



    return 0;

}
