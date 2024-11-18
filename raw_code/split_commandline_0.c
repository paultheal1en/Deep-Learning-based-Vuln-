int split_commandline(OptionParseContext *octx, int argc, char *argv[],

                      const OptionDef *options,

                      const OptionGroupDef *groups)

{

    int optindex = 1;



    /* perform system-dependent conversions for arguments list */

    prepare_app_arguments(&argc, &argv);



    init_parse_context(octx, groups);

    av_log(NULL, AV_LOG_DEBUG, "Splitting the commandline.\n");



    while (optindex < argc) {

        const char *opt = argv[optindex++], *arg;

        const OptionDef *po;

        int ret;



        av_log(NULL, AV_LOG_DEBUG, "Reading option '%s' ...", opt);



        /* unnamed group separators, e.g. output filename */

        if (opt[0] != '-' || !opt[1]) {

            finish_group(octx, 0, opt);

            av_log(NULL, AV_LOG_DEBUG, " matched as %s.\n", groups[0].name);

            continue;

        }

        opt++;



#define GET_ARG(arg)                                                           \

do {                                                                           \

    arg = argv[optindex++];                                                    \

    if (!arg) {                                                                \

        av_log(NULL, AV_LOG_ERROR, "Missing argument for option '%s'.\n", opt);\

        return AVERROR(EINVAL);                                                \

    }                                                                          \

} while (0)



        /* named group separators, e.g. -i */

        if ((ret = match_group_separator(groups, opt)) >= 0) {

            GET_ARG(arg);

            finish_group(octx, ret, arg);

            av_log(NULL, AV_LOG_DEBUG, " matched as %s with argument '%s'.\n",

                   groups[ret].name, arg);

            continue;

        }



        /* normal options */

        po = find_option(options, opt);

        if (po->name) {

            if (po->flags & OPT_EXIT) {

                /* optional argument, e.g. -h */

                arg = argv[optindex++];

            } else if (po->flags & HAS_ARG) {

                GET_ARG(arg);

            } else {

                arg = "1";

            }



            add_opt(octx, po, opt, arg);

            av_log(NULL, AV_LOG_DEBUG, " matched as option '%s' (%s) with "

                   "argument '%s'.\n", po->name, po->help, arg);

            continue;

        }



        /* AVOptions */

        if (argv[optindex]) {

            ret = opt_default(NULL, opt, argv[optindex]);

            if (ret >= 0) {

                av_log(NULL, AV_LOG_DEBUG, " matched as AVOption '%s' with "

                       "argument '%s'.\n", opt, argv[optindex]);

                optindex++;

                continue;

            } else if (ret != AVERROR_OPTION_NOT_FOUND) {

                av_log(NULL, AV_LOG_ERROR, "Error parsing option '%s' "

                       "with argument '%s'.\n", opt, argv[optindex]);

                return ret;

            }

        }



        /* boolean -nofoo options */

        if (opt[0] == 'n' && opt[1] == 'o' &&

            (po = find_option(options, opt + 2)) &&

            po->name && po->flags & OPT_BOOL) {

            add_opt(octx, po, opt, "0");

            av_log(NULL, AV_LOG_DEBUG, " matched as option '%s' (%s) with "

                   "argument 0.\n", po->name, po->help);

            continue;

        }



        av_log(NULL, AV_LOG_ERROR, "Unrecognized option '%s'.\n", opt);

        return AVERROR_OPTION_NOT_FOUND;

    }



    if (octx->cur_group.nb_opts || codec_opts || format_opts)

        av_log(NULL, AV_LOG_WARNING, "Trailing options were found on the "

               "commandline.\n");



    av_log(NULL, AV_LOG_DEBUG, "Finished splitting the commandline.\n");



    return 0;

}
