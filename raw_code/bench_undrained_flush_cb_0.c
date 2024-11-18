static void bench_undrained_flush_cb(void *opaque, int ret)

{

    if (ret < 0) {

        error_report("Failed flush request: %s\n", strerror(-ret));

        exit(EXIT_FAILURE);

    }

}
