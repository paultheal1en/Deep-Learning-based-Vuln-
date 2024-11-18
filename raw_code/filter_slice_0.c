static int filter_slice(AVFilterContext *ctx, void *arg, int jobnr,

                        int nb_jobs)

{

    TransContext *s = ctx->priv;

    ThreadData *td = arg;

    AVFrame *out = td->out;

    AVFrame *in = td->in;

    int plane;



    for (plane = 0; out->data[plane]; plane++) {

        int hsub    = plane == 1 || plane == 2 ? s->hsub : 0;

        int vsub    = plane == 1 || plane == 2 ? s->vsub : 0;

        int pixstep = s->pixsteps[plane];

        int inh     = AV_CEIL_RSHIFT(in->height, vsub);

        int outw    = AV_CEIL_RSHIFT(out->width,  hsub);

        int outh    = AV_CEIL_RSHIFT(out->height, vsub);

        int start   = (outh *  jobnr   ) / nb_jobs;

        int end     = (outh * (jobnr+1)) / nb_jobs;

        uint8_t *dst, *src;

        int dstlinesize, srclinesize;

        int x, y;



        dstlinesize = out->linesize[plane];

        dst         = out->data[plane] + start * dstlinesize;

        src         = in->data[plane];

        srclinesize = in->linesize[plane];



        if (s->dir & 1) {

            src         += in->linesize[plane] * (inh - 1);

            srclinesize *= -1;

        }



        if (s->dir & 2) {

            dst          = out->data[plane] + dstlinesize * (outh - start - 1);

            dstlinesize *= -1;

        }



        for (y = start; y < end - 7; y += 8) {

            for (x = 0; x < outw - 7; x += 8) {

                s->transpose_8x8(src + x * srclinesize + y * pixstep,

                                 srclinesize,

                                 dst + (y - start) * dstlinesize + x * pixstep,

                                 dstlinesize);

            }

            if (outw - x > 0 && end - y > 0)

                s->transpose_block(src + x * srclinesize + y * pixstep,

                                   srclinesize,

                                   dst + (y - start) * dstlinesize + x * pixstep,

                                   dstlinesize, outw - x, end - y);

        }



        if (end - y > 0)

            s->transpose_block(src + 0 * srclinesize + y * pixstep,

                               srclinesize,

                               dst + (y - start) * dstlinesize + 0 * pixstep,

                               dstlinesize, outw, end - y);

    }



    return 0;

}
