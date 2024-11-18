static void test_none(void)

{

    struct qdist dist;

    char *pr;



    qdist_init(&dist);



    g_assert(isnan(qdist_avg(&dist)));

    g_assert(isnan(qdist_xmin(&dist)));

    g_assert(isnan(qdist_xmax(&dist)));



    pr = qdist_pr_plain(&dist, 0);

    g_assert(pr == NULL);



    pr = qdist_pr_plain(&dist, 2);

    g_assert(pr == NULL);



    qdist_destroy(&dist);

}
