static inline int coef_test_compression(int coef)

{

    int tmp = coef >> 2;

    int res = ff_ctz(tmp);

    if (res > 1)

        return 1;       /* ...00 ->  compressable    */

    else if (res == 1)

        return 0;       /* ...10 ->  uncompressable  */

    else if (ff_ctz(tmp >> 1) > 0)

        return 0;       /* ...0 1 -> uncompressable  */

    else

        return 1;       /* ...1 1 -> compressable    */

}
