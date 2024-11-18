static int alloc_f(int argc, char **argv)

{

    int64_t offset;

    int nb_sectors, remaining;

    char s1[64];

    int num, sum_alloc;

    int ret;



    offset = cvtnum(argv[1]);

    if (offset & 0x1ff) {

        printf("offset %" PRId64 " is not sector aligned\n",

               offset);

        return 0;

    }



    if (argc == 3) {

        nb_sectors = cvtnum(argv[2]);

    } else {

        nb_sectors = 1;

    }



    remaining = nb_sectors;

    sum_alloc = 0;

    while (remaining) {

        ret = bdrv_is_allocated(bs, offset >> 9, nb_sectors, &num);

        remaining -= num;

        if (ret) {

            sum_alloc += num;

        }

    }



    cvtstr(offset, s1, sizeof(s1));



    printf("%d/%d sectors allocated at offset %s\n",

           sum_alloc, nb_sectors, s1);

    return 0;

}
