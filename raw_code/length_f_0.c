length_f(int argc, char **argv)

{

        int64_t size;

	char s1[64];



	size = bdrv_getlength(bs);

	if (size < 0) {

		printf("getlength: %s", strerror(size));

		return 0;

	}



	cvtstr(size, s1, sizeof(s1));

	printf("%s\n", s1);

	return 0;

}
