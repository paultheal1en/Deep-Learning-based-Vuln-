void av_opt_set_defaults2(void *s, int mask, int flags)

{

#endif

    const AVOption *opt = NULL;

    while ((opt = av_opt_next(s, opt)) != NULL) {

#if FF_API_OLD_AVOPTIONS

        if ((opt->flags & mask) != flags)

            continue;

#endif

        switch (opt->type) {

            case AV_OPT_TYPE_CONST:

                /* Nothing to be done here */

            break;

            case AV_OPT_TYPE_FLAGS:

            case AV_OPT_TYPE_INT:

            case AV_OPT_TYPE_INT64:

                av_opt_set_int(s, opt->name, opt->default_val.i64, 0);

            break;

            case AV_OPT_TYPE_DOUBLE:

            case AV_OPT_TYPE_FLOAT: {

                double val;

                val = opt->default_val.dbl;

                av_opt_set_double(s, opt->name, val, 0);

            }

            break;

            case AV_OPT_TYPE_RATIONAL: {

                AVRational val;

                val = av_d2q(opt->default_val.dbl, INT_MAX);

                av_opt_set_q(s, opt->name, val, 0);

            }

            break;

            case AV_OPT_TYPE_STRING:

            case AV_OPT_TYPE_IMAGE_SIZE:

            case AV_OPT_TYPE_PIXEL_FMT:

            case AV_OPT_TYPE_SAMPLE_FMT:

                av_opt_set(s, opt->name, opt->default_val.str, 0);

                break;

            case AV_OPT_TYPE_BINARY:

                /* Cannot set default for binary */

            break;

            default:

                av_log(s, AV_LOG_DEBUG, "AVOption type %d of option %s not implemented yet\n", opt->type, opt->name);

        }

    }

}
