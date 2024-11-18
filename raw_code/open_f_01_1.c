static int open_f(BlockBackend *blk, int argc, char **argv)

{

    int flags = BDRV_O_UNMAP;

    int readonly = 0;

    bool writethrough = true;

    int c;

    QemuOpts *qopts;

    QDict *opts;

    bool force_share = false;



    while ((c = getopt(argc, argv, "snro:kt:d:U")) != -1) {

        switch (c) {

        case 's':

            flags |= BDRV_O_SNAPSHOT;

            break;

        case 'n':

            flags |= BDRV_O_NOCACHE;

            writethrough = false;

            break;

        case 'r':

            readonly = 1;

            break;

        case 'k':

            flags |= BDRV_O_NATIVE_AIO;

            break;

        case 't':

            if (bdrv_parse_cache_mode(optarg, &flags, &writethrough) < 0) {

                error_report("Invalid cache option: %s", optarg);

                qemu_opts_reset(&empty_opts);

                return 0;

            }

            break;

        case 'd':

            if (bdrv_parse_discard_flags(optarg, &flags) < 0) {

                error_report("Invalid discard option: %s", optarg);

                qemu_opts_reset(&empty_opts);

                return 0;

            }

            break;

        case 'o':

            if (imageOpts) {

                printf("--image-opts and 'open -o' are mutually exclusive\n");

                qemu_opts_reset(&empty_opts);

                return 0;

            }

            if (!qemu_opts_parse_noisily(&empty_opts, optarg, false)) {

                qemu_opts_reset(&empty_opts);

                return 0;

            }

            break;

        case 'U':

            force_share = true;

            break;

        default:

            qemu_opts_reset(&empty_opts);

            return qemuio_command_usage(&open_cmd);

        }

    }



    if (!readonly) {

        flags |= BDRV_O_RDWR;

    }



    if (imageOpts && (optind == argc - 1)) {

        if (!qemu_opts_parse_noisily(&empty_opts, argv[optind], false)) {

            qemu_opts_reset(&empty_opts);

            return 0;

        }

        optind++;

    }



    qopts = qemu_opts_find(&empty_opts, NULL);

    opts = qopts ? qemu_opts_to_qdict(qopts, NULL) : NULL;

    qemu_opts_reset(&empty_opts);



    if (optind == argc - 1) {

        return openfile(argv[optind], flags, writethrough, force_share, opts);

    } else if (optind == argc) {

        return openfile(NULL, flags, writethrough, force_share, opts);

    } else {

        QDECREF(opts);

        return qemuio_command_usage(&open_cmd);

    }

}
