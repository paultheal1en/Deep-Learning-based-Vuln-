create_iovec(QEMUIOVector *qiov, char **argv, int nr_iov, int pattern)

{

    size_t *sizes = calloc(nr_iov, sizeof(size_t));

    size_t count = 0;

    void *buf = NULL;

    void *p;

    int i;



    for (i = 0; i < nr_iov; i++) {

        char *arg = argv[i];

        int64_t len;



        len = cvtnum(arg);

        if (len < 0) {

            printf("non-numeric length argument -- %s\n", arg);

            goto fail;

        }



        /* should be SIZE_T_MAX, but that doesn't exist */

        if (len > INT_MAX) {

            printf("too large length argument -- %s\n", arg);

            goto fail;

        }



        if (len & 0x1ff) {

            printf("length argument %" PRId64

                   " is not sector aligned\n", len);

            goto fail;

        }



        sizes[i] = len;

        count += len;

    }



    qemu_iovec_init(qiov, nr_iov);



    buf = p = qemu_io_alloc(count, pattern);



    for (i = 0; i < nr_iov; i++) {

        qemu_iovec_add(qiov, p, sizes[i]);

        p += sizes[i];

    }



fail:

    free(sizes);

    return buf;

}
