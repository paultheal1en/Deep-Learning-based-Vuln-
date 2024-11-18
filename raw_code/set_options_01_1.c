static inline int set_options(AVFilterContext *ctx, const char *args)

{

    HueContext *hue = ctx->priv;

    int n, ret;

    char c1 = 0, c2 = 0;

    char   *old_hue_expr,  *old_hue_deg_expr,  *old_saturation_expr;

    AVExpr *old_hue_pexpr, *old_hue_deg_pexpr, *old_saturation_pexpr;



    if (args) {

        /* named options syntax */

        if (strchr(args, '=')) {

            old_hue_expr        = hue->hue_expr;

            old_hue_deg_expr    = hue->hue_deg_expr;

            old_saturation_expr = hue->saturation_expr;



            old_hue_pexpr        = hue->hue_pexpr;

            old_hue_deg_pexpr    = hue->hue_deg_pexpr;

            old_saturation_pexpr = hue->saturation_pexpr;



            hue->hue_expr     = NULL;

            hue->hue_deg_expr = NULL;

            hue->saturation_expr = NULL;



            if ((ret = av_set_options_string(hue, args, "=", ":")) < 0)

                return ret;

            if (hue->hue_expr && hue->hue_deg_expr) {

                av_log(ctx, AV_LOG_ERROR,

                       "H and h options are incompatible and cannot be specified "

                       "at the same time\n");

                hue->hue_expr     = old_hue_expr;

                hue->hue_deg_expr = old_hue_deg_expr;



                return AVERROR(EINVAL);

            }



            /*

             * if both 'H' and 'h' options have not been specified, restore the

             * old values

             */

            if (!hue->hue_expr && !hue->hue_deg_expr) {

                hue->hue_expr     = old_hue_expr;

                hue->hue_deg_expr = old_hue_deg_expr;

            }



            if (hue->hue_deg_expr)

                PARSE_EXPRESSION(hue_deg, h);

            if (hue->hue_expr)

                PARSE_EXPRESSION(hue, H);

            if (hue->saturation_expr)

                PARSE_EXPRESSION(saturation, s);



            hue->flat_syntax = 0;



            av_log(ctx, AV_LOG_VERBOSE,

                   "H_expr:%s h_deg_expr:%s s_expr:%s\n",

                   hue->hue_expr, hue->hue_deg_expr, hue->saturation_expr);



        /* compatibility h:s syntax */

        } else {

            n = sscanf(args, "%f%c%f%c", &hue->hue_deg, &c1, &hue->saturation, &c2);

            if (n != 1 && (n != 3 || c1 != ':')) {

                av_log(ctx, AV_LOG_ERROR,

                       "Invalid syntax for argument '%s': "

                       "must be in the form 'hue[:saturation]'\n", args);

                return AVERROR(EINVAL);

            }



            if (hue->saturation < SAT_MIN_VAL || hue->saturation > SAT_MAX_VAL) {

                av_log(ctx, AV_LOG_ERROR,

                       "Invalid value for saturation %0.1f: "

                       "must be included between range %d and +%d\n",

                       hue->saturation, SAT_MIN_VAL, SAT_MAX_VAL);

                return AVERROR(EINVAL);

            }



            hue->hue = hue->hue_deg * M_PI / 180;

            hue->flat_syntax = 1;



            av_log(ctx, AV_LOG_VERBOSE,

                   "H:%0.1f h:%0.1f s:%0.1f\n",

                   hue->hue, hue->hue_deg, hue->saturation);

        }

    }



    compute_sin_and_cos(hue);



    return 0;

}
