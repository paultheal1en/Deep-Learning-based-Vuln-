static void qht_do_test(unsigned int mode, size_t init_entries)
{
    qht_init(&ht, 0, mode);
    insert(0, N);
    check(0, N, true);
    check_n(N);
    check(-N, -1, false);
    iter_check(N);
    rm(101, 102);
    check_n(N - 1);
    insert(N, N * 2);
    check_n(N + N - 1);
    rm(N, N * 2);
    check_n(N - 1);
    insert(101, 102);
    check_n(N);
    rm(10, 200);
    check_n(N - 190);
    insert(150, 200);
    check_n(N - 190 + 50);
    insert(10, 150);
    check_n(N);
    rm(1, 2);
    check_n(N - 1);
    qht_reset_size(&ht, 0);
    check(0, N, false);
    qht_destroy(&ht);
}