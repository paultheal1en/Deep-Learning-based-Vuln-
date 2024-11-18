static int pc_rec_cmp(const void *p1, const void *p2)

{

    PCRecord *r1 = *(PCRecord **)p1;

    PCRecord *r2 = *(PCRecord **)p2;

    if (r1->count < r2->count)

        return 1;

    else if (r1->count == r2->count)

        return 0;

    else

        return -1;

}
