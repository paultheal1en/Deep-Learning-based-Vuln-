static int discard_f(BlockBackend *blk, int argc, char **argv)

{

    struct timeval t1, t2;

    int Cflag = 0, qflag = 0;

    int c, ret;

    int64_t offset, count;



    while ((c = getopt(argc, argv, "Cq")) != -1) {

        switch (c) {

        case 'C':

            Cflag = 1;

            break;

        case 'q':

            qflag = 1;

            break;

        default:

            return qemuio_command_usage(&discard_cmd);

        }

    }



    if (optind != argc - 2) {

        return qemuio_command_usage(&discard_cmd);

    }



    offset = cvtnum(argv[optind]);

    if (offset < 0) {

        print_cvtnum_err(offset, argv[optind]);

        return 0;

    }



    optind++;

    count = cvtnum(argv[optind]);

    if (count < 0) {

        print_cvtnum_err(count, argv[optind]);

        return 0;

    } else if (count >> BDRV_SECTOR_BITS > INT_MAX) {

        printf("length cannot exceed %"PRIu64", given %s\n",

               (uint64_t)INT_MAX << BDRV_SECTOR_BITS,

               argv[optind]);

        return 0;

    }



    gettimeofday(&t1, NULL);

    ret = blk_discard(blk, offset >> BDRV_SECTOR_BITS,

                      count >> BDRV_SECTOR_BITS);

    gettimeofday(&t2, NULL);



    if (ret < 0) {

        printf("discard failed: %s\n", strerror(-ret));

        goto out;

    }



    /* Finally, report back -- -C gives a parsable format */

    if (!qflag) {

        t2 = tsub(t2, t1);

        print_report("discard", &t2, offset, count, count, 1, Cflag);

    }



out:

    return 0;

}
