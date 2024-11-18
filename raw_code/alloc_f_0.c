alloc_f(int argc, char **argv)

{

	int64_t offset;

	int nb_sectors;

	char s1[64];

	int num;

	int ret;

	const char *retstr;



	offset = cvtnum(argv[1]);

	if (offset & 0x1ff) {

		printf("offset %lld is not sector aligned\n",

			(long long)offset);

		return 0;

	}



	if (argc == 3)

		nb_sectors = cvtnum(argv[2]);

	else

		nb_sectors = 1;



	ret = bdrv_is_allocated(bs, offset >> 9, nb_sectors, &num);



	cvtstr(offset, s1, sizeof(s1));



	retstr = ret ? "allocated" : "not allocated";

	if (nb_sectors == 1)

		printf("sector %s at offset %s\n", retstr, s1);

	else

		printf("%d/%d sectors %s at offset %s\n",

			num, nb_sectors, retstr, s1);

	return 0;

}
