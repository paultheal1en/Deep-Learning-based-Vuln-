static void test_rtas_get_time_of_day(void)

{

    QOSState *qs;

    struct tm tm;

    uint32_t ns;

    uint64_t ret;

    time_t t1, t2;



    qs = qtest_spapr_boot("-machine pseries");

    g_assert(qs != NULL);



    t1 = time(NULL);

    ret = qrtas_get_time_of_day(qs->alloc, &tm, &ns);

    g_assert_cmpint(ret, ==, 0);

    t2 = mktimegm(&tm);

    g_assert(t2 - t1 < 5); /* 5 sec max to run the test */



    qtest_shutdown(qs);

}
