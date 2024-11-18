create_iovec(QEMUIOVector *qiov, char **argv, int nr_iov, int pattern)

{

	size_t *sizes = calloc(nr_iov, sizeof(size_t));

	size_t count = 0;

	void *buf, *p;

	int i;



	for (i = 0; i < nr_iov; i++) {

		char *arg = argv[i];

		long long len;



		len = cvtnum(arg);

		if (len < 0) {

			printf("non-numeric length argument -- %s\n", arg);

			return NULL;

		}



		/* should be SIZE_T_MAX, but that doesn't exist */

		if (len > UINT_MAX) {

			printf("too large length argument -- %s\n", arg);

			return NULL;

		}



		if (len & 0x1ff) {

			printf("length argument %lld is not sector aligned\n",

				len);

			return NULL;

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



	free(sizes);

	return buf;

}
