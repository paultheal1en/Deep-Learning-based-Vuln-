static int img_check(int argc, char **argv)

{

    int c, ret;

    OutputFormat output_format = OFORMAT_HUMAN;

    const char *filename, *fmt, *output, *cache;

    BlockBackend *blk;

    BlockDriverState *bs;

    int fix = 0;

    int flags = BDRV_O_FLAGS | BDRV_O_CHECK;

    ImageCheck *check;

    bool quiet = false;



    fmt = NULL;

    output = NULL;

    cache = BDRV_DEFAULT_CACHE;

    for(;;) {

        int option_index = 0;

        static const struct option long_options[] = {

            {"help", no_argument, 0, 'h'},

            {"format", required_argument, 0, 'f'},

            {"repair", required_argument, 0, 'r'},

            {"output", required_argument, 0, OPTION_OUTPUT},

            {0, 0, 0, 0}

        };

        c = getopt_long(argc, argv, "hf:r:T:q",

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

        case 'r':

            flags |= BDRV_O_RDWR;



            if (!strcmp(optarg, "leaks")) {

                fix = BDRV_FIX_LEAKS;

            } else if (!strcmp(optarg, "all")) {

                fix = BDRV_FIX_LEAKS | BDRV_FIX_ERRORS;

            } else {

                error_exit("Unknown option value for -r "

                           "(expecting 'leaks' or 'all'): %s", optarg);

            }

            break;

        case OPTION_OUTPUT:

            output = optarg;

            break;

        case 'T':

            cache = optarg;

            break;

        case 'q':

            quiet = true;

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



    ret = bdrv_parse_cache_flags(cache, &flags);

    if (ret < 0) {

        error_report("Invalid source cache option: %s", cache);

        return 1;

    }



    blk = img_open("image", filename, fmt, flags, true, quiet);

    if (!blk) {

        return 1;

    }

    bs = blk_bs(blk);



    check = g_new0(ImageCheck, 1);

    ret = collect_image_check(bs, check, filename, fmt, fix);



    if (ret == -ENOTSUP) {

        error_report("This image format does not support checks");

        ret = 63;

        goto fail;

    }



    if (check->corruptions_fixed || check->leaks_fixed) {

        int corruptions_fixed, leaks_fixed;



        leaks_fixed         = check->leaks_fixed;

        corruptions_fixed   = check->corruptions_fixed;



        if (output_format == OFORMAT_HUMAN) {

            qprintf(quiet,

                    "The following inconsistencies were found and repaired:\n\n"

                    "    %" PRId64 " leaked clusters\n"

                    "    %" PRId64 " corruptions\n\n"

                    "Double checking the fixed image now...\n",

                    check->leaks_fixed,

                    check->corruptions_fixed);

        }



        ret = collect_image_check(bs, check, filename, fmt, 0);



        check->leaks_fixed          = leaks_fixed;

        check->corruptions_fixed    = corruptions_fixed;

    }



    switch (output_format) {

    case OFORMAT_HUMAN:

        dump_human_image_check(check, quiet);

        break;

    case OFORMAT_JSON:

        dump_json_image_check(check, quiet);

        break;

    }



    if (ret || check->check_errors) {

        ret = 1;

        goto fail;

    }



    if (check->corruptions) {

        ret = 2;

    } else if (check->leaks) {

        ret = 3;

    } else {

        ret = 0;

    }



fail:

    qapi_free_ImageCheck(check);

    blk_unref(blk);

    return ret;

}
