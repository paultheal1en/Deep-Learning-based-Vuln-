static int config_input_overlay(AVFilterLink *inlink)

{

    AVFilterContext *ctx  = inlink->dst;

    OverlayContext  *over = inlink->dst->priv;

    char *expr;

    double var_values[VAR_VARS_NB], res;

    int ret;

    const AVPixFmtDescriptor *pix_desc = av_pix_fmt_desc_get(inlink->format);



    av_image_fill_max_pixsteps(over->overlay_pix_step, NULL, pix_desc);



    /* Finish the configuration by evaluating the expressions

       now when both inputs are configured. */

    var_values[VAR_MAIN_W   ] = var_values[VAR_MW] = ctx->inputs[MAIN   ]->w;

    var_values[VAR_MAIN_H   ] = var_values[VAR_MH] = ctx->inputs[MAIN   ]->h;

    var_values[VAR_OVERLAY_W] = var_values[VAR_OW] = ctx->inputs[OVERLAY]->w;

    var_values[VAR_OVERLAY_H] = var_values[VAR_OH] = ctx->inputs[OVERLAY]->h;



    if ((ret = av_expr_parse_and_eval(&res, (expr = over->x_expr), var_names, var_values,

                                      NULL, NULL, NULL, NULL, NULL, 0, ctx)) < 0)

        goto fail;

    over->x = res;

    if ((ret = av_expr_parse_and_eval(&res, (expr = over->y_expr), var_names, var_values,

                                      NULL, NULL, NULL, NULL, NULL, 0, ctx)))

        goto fail;

    over->y = res;

    /* x may depend on y */

    if ((ret = av_expr_parse_and_eval(&res, (expr = over->x_expr), var_names, var_values,

                                      NULL, NULL, NULL, NULL, NULL, 0, ctx)) < 0)

        goto fail;

    over->x = res;



    over->overlay_is_packed_rgb =

        ff_fill_rgba_map(over->overlay_rgba_map, inlink->format) >= 0;

    over->overlay_has_alpha = ff_fmt_is_in(inlink->format, alpha_pix_fmts);



    av_log(ctx, AV_LOG_VERBOSE,

           "main w:%d h:%d fmt:%s overlay x:%d y:%d w:%d h:%d fmt:%s\n",

           ctx->inputs[MAIN]->w, ctx->inputs[MAIN]->h,

           av_get_pix_fmt_name(ctx->inputs[MAIN]->format),

           over->x, over->y,

           ctx->inputs[OVERLAY]->w, ctx->inputs[OVERLAY]->h,

           av_get_pix_fmt_name(ctx->inputs[OVERLAY]->format));



    if (over->x < 0 || over->y < 0 ||

        over->x + var_values[VAR_OVERLAY_W] > var_values[VAR_MAIN_W] ||

        over->y + var_values[VAR_OVERLAY_H] > var_values[VAR_MAIN_H]) {

        av_log(ctx, AV_LOG_ERROR,

               "Overlay area (%d,%d)<->(%d,%d) not within the main area (0,0)<->(%d,%d) or zero-sized\n",

               over->x, over->y,

               (int)(over->x + var_values[VAR_OVERLAY_W]),

               (int)(over->y + var_values[VAR_OVERLAY_H]),

               (int)var_values[VAR_MAIN_W], (int)var_values[VAR_MAIN_H]);

        return AVERROR(EINVAL);

    }

    return 0;



fail:

    av_log(NULL, AV_LOG_ERROR,

           "Error when evaluating the expression '%s'\n", expr);

    return ret;

}
