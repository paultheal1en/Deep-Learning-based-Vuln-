static int img_info(int argc, char **argv)

{

    int c;

    OutputFormat output_format = OFORMAT_HUMAN;

    const char *filename, *fmt, *output;

    BlockDriverState *bs;

    ImageInfo *info;



    fmt = NULL;

    output = NULL;

    for(;;) {

        int option_index = 0;

        static const struct option long_options[] = {

            {"help", no_argument, 0, 'h'},

            {"format", required_argument, 0, 'f'},

            {"output", required_argument, 0, OPTION_OUTPUT},

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

        }

    }

    if (optind >= argc) {

        help();

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



    bs = bdrv_new_open(filename, fmt, BDRV_O_FLAGS | BDRV_O_NO_BACKING);

    if (!bs) {

        return 1;

    }



    info = g_new0(ImageInfo, 1);

    collect_image_info(bs, info, filename, fmt);



    switch (output_format) {

    case OFORMAT_HUMAN:

        dump_human_image_info(info);

        dump_snapshots(bs);

        break;

    case OFORMAT_JSON:

        collect_snapshots(bs, info);

        dump_json_image_info(info);

        break;

    }



    qapi_free_ImageInfo(info);

    bdrv_delete(bs);

    return 0;

}
