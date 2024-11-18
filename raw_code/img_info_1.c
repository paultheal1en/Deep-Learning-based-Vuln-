static int img_info(int argc, char **argv)

{

    int c;

    OutputFormat output_format = OFORMAT_HUMAN;

    bool chain = false;

    const char *filename, *fmt, *output;

    ImageInfoList *list;

    bool image_opts = false;



    fmt = NULL;

    output = NULL;

    for(;;) {

        int option_index = 0;

        static const struct option long_options[] = {

            {"help", no_argument, 0, 'h'},

            {"format", required_argument, 0, 'f'},

            {"output", required_argument, 0, OPTION_OUTPUT},

            {"backing-chain", no_argument, 0, OPTION_BACKING_CHAIN},

            {"object", required_argument, 0, OPTION_OBJECT},

            {"image-opts", no_argument, 0, OPTION_IMAGE_OPTS},

            {0, 0, 0, 0}

        };

        c = getopt_long(argc, argv, "f:h",

                        long_options, &option_index);

        if (c == -1) {

            break;

        }

        switch(c) {

        case '?':

        case 'h':

            help();

            break;

        case 'f':

            fmt = optarg;

            break;

        case OPTION_OUTPUT:

            output = optarg;

            break;

        case OPTION_BACKING_CHAIN:

            chain = true;

            break;

        case OPTION_OBJECT: {

            QemuOpts *opts;

            opts = qemu_opts_parse_noisily(&qemu_object_opts,

                                           optarg, true);

            if (!opts) {

                return 1;

            }

        }   break;

        case OPTION_IMAGE_OPTS:

            image_opts = true;

            break;

        }

    }

    if (optind != argc - 1) {

        error_exit("Expecting one image file name");

    }

    filename = argv[optind++];



    if (output && !strcmp(output, "json")) {

        output_format = OFORMAT_JSON;

    } else if (output && !strcmp(output, "human")) {

        output_format = OFORMAT_HUMAN;

    } else if (output) {

        error_report("--output must be used with human or json as argument.");

        return 1;

    }



    if (qemu_opts_foreach(&qemu_object_opts,

                          user_creatable_add_opts_foreach,

                          NULL, NULL)) {

        return 1;

    }



    list = collect_image_info_list(image_opts, filename, fmt, chain);

    if (!list) {

        return 1;

    }



    switch (output_format) {

    case OFORMAT_HUMAN:

        dump_human_image_info_list(list);

        break;

    case OFORMAT_JSON:

        if (chain) {

            dump_json_image_info_list(list);

        } else {

            dump_json_image_info(list->value);

        }

        break;

    }



    qapi_free_ImageInfoList(list);

    return 0;

}
