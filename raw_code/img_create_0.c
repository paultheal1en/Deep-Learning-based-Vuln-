static int img_create(int argc, char **argv)

{

    int c, ret, flags;

    const char *fmt = "raw";

    const char *base_fmt = NULL;

    const char *filename;

    const char *base_filename = NULL;

    BlockDriver *drv;

    QEMUOptionParameter *param = NULL;

    char *options = NULL;



    flags = 0;

    for(;;) {

        c = getopt(argc, argv, "F:b:f:he6o:");

        if (c == -1)

            break;

        switch(c) {

        case 'h':

            help();

            break;

        case 'F':

            base_fmt = optarg;

            break;

        case 'b':

            base_filename = optarg;

            break;

        case 'f':

            fmt = optarg;

            break;

        case 'e':

            flags |= BLOCK_FLAG_ENCRYPT;

            break;

        case '6':

            flags |= BLOCK_FLAG_COMPAT6;

            break;

        case 'o':

            options = optarg;

            break;

        }

    }



    /* Find driver and parse its options */

    drv = bdrv_find_format(fmt);

    if (!drv)

        error("Unknown file format '%s'", fmt);



    if (options && !strcmp(options, "?")) {

        print_option_help(drv->create_options);

        return 0;

    }



    if (options) {

        param = parse_option_parameters(options, drv->create_options, param);

        if (param == NULL) {

            error("Invalid options for file format '%s'.", fmt);

        }

    } else {

        param = parse_option_parameters("", drv->create_options, param);

    }



    /* Get the filename */

    if (optind >= argc)

        help();

    filename = argv[optind++];



    /* Add size to parameters */

    if (optind < argc) {

        set_option_parameter(param, BLOCK_OPT_SIZE, argv[optind++]);

    }



    /* Add old-style options to parameters */

    add_old_style_options(fmt, param, flags, base_filename, base_fmt);



    // The size for the image must always be specified, with one exception:

    // If we are using a backing file, we can obtain the size from there

    if (get_option_parameter(param, BLOCK_OPT_SIZE)->value.n == 0) {



        QEMUOptionParameter *backing_file =

            get_option_parameter(param, BLOCK_OPT_BACKING_FILE);

        QEMUOptionParameter *backing_fmt =

            get_option_parameter(param, BLOCK_OPT_BACKING_FMT);



        if (backing_file && backing_file->value.s) {

            BlockDriverState *bs;

            uint64_t size;

            const char *fmt = NULL;

            char buf[32];



            if (backing_fmt && backing_fmt->value.s) {

                 if (bdrv_find_format(backing_fmt->value.s)) {

                     fmt = backing_fmt->value.s;

                } else {

                     error("Unknown backing file format '%s'",

                        backing_fmt->value.s);

                }

            }



            bs = bdrv_new_open(backing_file->value.s, fmt);

            bdrv_get_geometry(bs, &size);

            size *= 512;

            bdrv_delete(bs);



            snprintf(buf, sizeof(buf), "%" PRId64, size);

            set_option_parameter(param, BLOCK_OPT_SIZE, buf);

        } else {

            error("Image creation needs a size parameter");

        }

    }



    printf("Formatting '%s', fmt=%s ", filename, fmt);

    print_option_parameters(param);

    puts("");



    ret = bdrv_create(drv, filename, param);

    free_option_parameters(param);



    if (ret < 0) {

        if (ret == -ENOTSUP) {

            error("Formatting or formatting option not supported for file format '%s'", fmt);

        } else if (ret == -EFBIG) {

            error("The image size is too large for file format '%s'", fmt);

        } else {

            error("Error while formatting");

        }

    }

    return 0;

}
