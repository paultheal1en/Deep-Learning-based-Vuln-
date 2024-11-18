static void test_vector_dmul_scalar(const double *src0, const double *src1)

{

    LOCAL_ALIGNED_32(double, cdst, [LEN]);

    LOCAL_ALIGNED_32(double, odst, [LEN]);

    int i;



    declare_func(void, double *dst, const double *src, double mul, int len);



    call_ref(cdst, src0, src1[0], LEN);

    call_new(odst, src0, src1[0], LEN);

    for (i = 0; i < LEN; i++) {

        if (!double_near_abs_eps(cdst[i], odst[i], DBL_EPSILON)) {

            fprintf(stderr, "%d: %- .12f - %- .12f = % .12g\n", i,

                    cdst[i], odst[i], cdst[i] - odst[i]);

            fail();

            break;

        }

    }

    bench_new(odst, src0, src1[0], LEN);

}
