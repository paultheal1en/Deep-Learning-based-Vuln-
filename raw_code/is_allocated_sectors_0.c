static int is_allocated_sectors(const uint8_t *buf, int n, int *pnum)

{

    int v, i;



    if (n <= 0) {

        *pnum = 0;

        return 0;

    }

    v = is_not_zero(buf, 512);

    for(i = 1; i < n; i++) {

        buf += 512;

        if (v != is_not_zero(buf, 512))

            break;

    }

    *pnum = i;

    return v;

}
