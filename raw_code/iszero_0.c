static int iszero(const int16_t *c, int sz)

{

    int n;



    for (n = 0; n < sz; n += 4)

        if (AV_RN32A(&c[n]))

            return 0;



    return 1;

}
