static int writev_f(int argc, char **argv)

{

    struct timeval t1, t2;

    int Cflag = 0, qflag = 0;

    int c, cnt;

    char *buf;

    int64_t offset;

    /* Some compilers get confused and warn if this is not initialized.  */

    int total = 0;

    int nr_iov;

    int pattern = 0xcd;

    QEMUIOVector qiov;



    while ((c = getopt(argc, argv, "CqP:")) != EOF) {

        switch (c) {

        case 'C':

            Cflag = 1;

            break;

        case 'q':

            qflag = 1;

            break;

        case 'P':

            pattern = parse_pattern(optarg);

            if (pattern < 0) {

                return 0;

            }

            break;

        default:

            return command_usage(&writev_cmd);

        }

    }



    if (optind > argc - 2) {

        return command_usage(&writev_cmd);

    }



    offset = cvtnum(argv[optind]);

    if (offset < 0) {

        printf("non-numeric length argument -- %s\n", argv[optind]);

        return 0;

    }

    optind++;



    if (offset & 0x1ff) {

        printf("offset %" PRId64 " is not sector aligned\n",

               offset);

        return 0;

    }



    nr_iov = argc - optind;

    buf = create_iovec(&qiov, &argv[optind], nr_iov, pattern);

    if (buf == NULL) {

        return 0;

    }



    gettimeofday(&t1, NULL);

    cnt = do_aio_writev(&qiov, offset, &total);

    gettimeofday(&t2, NULL);



    if (cnt < 0) {

        printf("writev failed: %s\n", strerror(-cnt));

        goto out;

    }



    if (qflag) {

        goto out;

    }



    /* Finally, report back -- -C gives a parsable format */

    t2 = tsub(t2, t1);

    print_report("wrote", &t2, offset, qiov.size, total, cnt, Cflag);

out:


    qemu_io_free(buf);

    return 0;

}