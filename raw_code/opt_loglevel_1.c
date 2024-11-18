int opt_loglevel(void *optctx, const char *opt, const char *arg)

{

    const struct { const char *name; int level; } log_levels[] = {

        { "quiet"  , AV_LOG_QUIET   },

        { "panic"  , AV_LOG_PANIC   },

        { "fatal"  , AV_LOG_FATAL   },

        { "error"  , AV_LOG_ERROR   },

        { "warning", AV_LOG_WARNING },

        { "info"   , AV_LOG_INFO    },

        { "verbose", AV_LOG_VERBOSE },

        { "debug"  , AV_LOG_DEBUG   },

    };

    char *tail;

    int level;

    int i;



    for (i = 0; i < FF_ARRAY_ELEMS(log_levels); i++) {

        if (!strcmp(log_levels[i].name, arg)) {

            av_log_set_level(log_levels[i].level);

            return 0;

        }

    }



    level = strtol(arg, &tail, 10);

    if (*tail) {

        av_log(NULL, AV_LOG_FATAL, "Invalid loglevel \"%s\". "

               "Possible levels are numbers or:\n", arg);

        for (i = 0; i < FF_ARRAY_ELEMS(log_levels); i++)

            av_log(NULL, AV_LOG_FATAL, "\"%s\"\n", log_levels[i].name);

        exit(1);

    }

    av_log_set_level(level);

    return 0;

}
