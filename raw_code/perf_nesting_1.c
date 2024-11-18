static void perf_nesting(void)

{

    unsigned int i, maxcycles, maxnesting;

    double duration;



    maxcycles = 10000;

    maxnesting = 1000;

    Coroutine *root;



    g_test_timer_start();

    for (i = 0; i < maxcycles; i++) {

        NestData nd = {

            .n_enter  = 0,

            .n_return = 0,

            .max      = maxnesting,

        };

        root = qemu_coroutine_create(nest);

        qemu_coroutine_enter(root, &nd);

    }

    duration = g_test_timer_elapsed();



    g_test_message("Nesting %u iterations of %u depth each: %f s\n",

        maxcycles, maxnesting, duration);

}
