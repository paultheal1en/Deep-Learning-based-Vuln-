truncate_f(int argc, char **argv)

{

	int64_t offset;

	int ret;



	offset = cvtnum(argv[1]);

	if (offset < 0) {

		printf("non-numeric truncate argument -- %s\n", argv[1]);

		return 0;

	}



	ret = bdrv_truncate(bs, offset);

	if (ret < 0) {

		printf("truncate: %s", strerror(ret));

		return 0;

	}



	return 0;

}
