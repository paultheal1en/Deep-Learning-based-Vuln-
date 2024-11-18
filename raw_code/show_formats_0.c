int show_formats(void *optctx, const char *opt, const char *arg)

{

    AVInputFormat *ifmt  = NULL;

    AVOutputFormat *ofmt = NULL;

    const char *last_name;



    printf("File formats:\n"

           " D. = Demuxing supported\n"

           " .E = Muxing supported\n"

           " --\n");

    last_name = "000";

    for (;;) {

        int decode = 0;

        int encode = 0;

        const char *name      = NULL;

        const char *long_name = NULL;



        while ((ofmt = av_oformat_next(ofmt))) {

            if ((name == NULL || strcmp(ofmt->name, name) < 0) &&

                strcmp(ofmt->name, last_name) > 0) {

                name      = ofmt->name;

                long_name = ofmt->long_name;

                encode    = 1;

            }

        }

        while ((ifmt = av_iformat_next(ifmt))) {

            if ((name == NULL || strcmp(ifmt->name, name) < 0) &&

                strcmp(ifmt->name, last_name) > 0) {

                name      = ifmt->name;

                long_name = ifmt->long_name;

                encode    = 0;

            }

            if (name && strcmp(ifmt->name, name) == 0)

                decode = 1;

        }

        if (name == NULL)

            break;

        last_name = name;



        printf(" %s%s %-15s %s\n",

               decode ? "D" : " ",

               encode ? "E" : " ",

               name,

            long_name ? long_name:" ");

    }

    return 0;

}
