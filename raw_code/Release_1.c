void Release(void *ctx)

{

    ContextInfo *ci;

    ci = (ContextInfo *) ctx;



    if (ci->cache) {

        imlib_context_set_image(ci->cache->image);

        imlib_free_image();

        av_free(ci->cache);

    }

    if (ctx) {

        if (ci->imageOverlaid) {

            imlib_context_set_image(ci->imageOverlaid);

            imlib_free_image();

        }

        ff_eval_free(ci->expr_x);

        ff_eval_free(ci->expr_y);

        ff_eval_free(ci->expr_R);

        ff_eval_free(ci->expr_G);

        ff_eval_free(ci->expr_B);

        sws_freeContext(ci->toRGB_convert_ctx);

        sws_freeContext(ci->fromRGB_convert_ctx);

        av_free(ctx);

    }

}
