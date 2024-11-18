static int img_check(int argc, char **argv)

{

    int c, ret;

    const char *filename, *fmt;

    BlockDriverState *bs;

    BdrvCheckResult result;

    int fix = 0;

    int flags = BDRV_O_FLAGS;



    fmt = NULL;

    for(;;) {

        c = getopt(argc, argv, "f:hr:");

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

                help();

            }

            break;

        }

    }

    if (optind >= argc) {

        help();

    }

    filename = argv[optind++];



    bs = bdrv_new_open(filename, fmt, flags);

    if (!bs) {

        return 1;

    }

    ret = bdrv_check(bs, &result, fix);



    if (ret == -ENOTSUP) {

        error_report("This image format does not support checks");

        bdrv_delete(bs);

        return 1;

    }



    if (result.corruptions_fixed || result.leaks_fixed) {

        printf("The following inconsistencies were found and repaired:\n\n"

               "    %d leaked clusters\n"

               "    %d corruptions\n\n"

               "Double checking the fixed image now...\n",

               result.leaks_fixed,

               result.corruptions_fixed);

        ret = bdrv_check(bs, &result, 0);

    }



    if (!(result.corruptions || result.leaks || result.check_errors)) {

        printf("No errors were found on the image.\n");

    } else {

        if (result.corruptions) {

            printf("\n%d errors were found on the image.\n"

                "Data may be corrupted, or further writes to the image "

                "may corrupt it.\n",

                result.corruptions);

        }



        if (result.leaks) {

            printf("\n%d leaked clusters were found on the image.\n"

                "This means waste of disk space, but no harm to data.\n",

                result.leaks);

        }



        if (result.check_errors) {

            printf("\n%d internal errors have occurred during the check.\n",

                result.check_errors);

        }

    }



    if (result.bfi.total_clusters != 0 && result.bfi.allocated_clusters != 0) {

        printf("%" PRId64 "/%" PRId64 "= %0.2f%% allocated, %0.2f%% fragmented\n",

        result.bfi.allocated_clusters, result.bfi.total_clusters,

        result.bfi.allocated_clusters * 100.0 / result.bfi.total_clusters,

        result.bfi.fragmented_clusters * 100.0 / result.bfi.allocated_clusters);

    }



    bdrv_delete(bs);



    if (ret < 0 || result.check_errors) {

        printf("\nAn error has occurred during the check: %s\n"

            "The check is not complete and may have missed error.\n",

            strerror(-ret));

        return 1;

    }



    if (result.corruptions) {

        return 2;

    } else if (result.leaks) {

        return 3;

    } else {

        return 0;

    }

}
