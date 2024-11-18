static SoftFloat sbr_sum_square_c(int (*x)[2], int n)

{

    SoftFloat ret;

    uint64_t accu = 0, round;

    int i, nz;

    unsigned u;



    for (i = 0; i < n; i += 2) {

        // Larger values are inavlid and could cause overflows of accu.

        av_assert2(FFABS(x[i + 0][0]) >> 29 == 0);

        accu += (int64_t)x[i + 0][0] * x[i + 0][0];

        av_assert2(FFABS(x[i + 0][1]) >> 29 == 0);

        accu += (int64_t)x[i + 0][1] * x[i + 0][1];

        av_assert2(FFABS(x[i + 1][0]) >> 29 == 0);

        accu += (int64_t)x[i + 1][0] * x[i + 1][0];

        av_assert2(FFABS(x[i + 1][1]) >> 29 == 0);

        accu += (int64_t)x[i + 1][1] * x[i + 1][1];

    }



    u = accu >> 32;

    if (u == 0) {

        nz = 1;

    } else {

        nz = -1;

        while (u < 0x80000000U) {

            u <<= 1;

            nz++;

        }

        nz = 32 - nz;

    }



    round = 1ULL << (nz-1);

    u = ((accu + round) >> nz);

    u >>= 1;

    ret = av_int2sf(u, 15 - nz);



    return ret;

}
