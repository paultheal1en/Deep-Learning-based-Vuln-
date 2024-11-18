static int set_params(AVFilterContext *ctx, const char *params)
{
    Frei0rContext *frei0r = ctx->priv;
    int i;
    for (i = 0; i < frei0r->plugin_info.num_params; i++) {
        f0r_param_info_t info;
        char *param;
        int ret;
        frei0r->get_param_info(&info, i);
        if (*params) {
            if (!(param = av_get_token(&params, "|")))
                return AVERROR(ENOMEM);
            params++;               /* skip ':' */
            ret = set_param(ctx, info, i, param);
            av_free(param);
            if (ret < 0)
                return ret;
        }
        av_log(ctx, AV_LOG_VERBOSE,
               "idx:%d name:'%s' type:%s explanation:'%s' ",
               i, info.name,
               info.type == F0R_PARAM_BOOL     ? "bool"     :
               info.type == F0R_PARAM_DOUBLE   ? "double"   :
               info.type == F0R_PARAM_COLOR    ? "color"    :
               info.type == F0R_PARAM_POSITION ? "position" :
               info.type == F0R_PARAM_STRING   ? "string"   : "unknown",
               info.explanation);
#ifdef DEBUG
        av_log(ctx, AV_LOG_DEBUG, "value:");
        switch (info.type) {
            void *v;
            double d;
            char s[128];
            f0r_param_color_t col;
            f0r_param_position_t pos;
        case F0R_PARAM_BOOL:
            v = &d;
            frei0r->get_param_value(frei0r->instance, v, i);
            av_log(ctx, AV_LOG_DEBUG, "%s", d >= 0.5 && d <= 1.0 ? "y" : "n");
            break;
        case F0R_PARAM_DOUBLE:
            v = &d;
            frei0r->get_param_value(frei0r->instance, v, i);
            av_log(ctx, AV_LOG_DEBUG, "%f", d);
            break;
        case F0R_PARAM_COLOR:
            v = &col;
            frei0r->get_param_value(frei0r->instance, v, i);
            av_log(ctx, AV_LOG_DEBUG, "%f/%f/%f", col.r, col.g, col.b);
            break;
        case F0R_PARAM_POSITION:
            v = &pos;
            frei0r->get_param_value(frei0r->instance, v, i);
            av_log(ctx, AV_LOG_DEBUG, "%f/%f", pos.x, pos.y);
            break;
        default: /* F0R_PARAM_STRING */
            v = s;
            frei0r->get_param_value(frei0r->instance, v, i);
            av_log(ctx, AV_LOG_DEBUG, "'%s'\n", s);
            break;
        }
#endif
        av_log(ctx, AV_LOG_VERBOSE, "\n");
    }
}