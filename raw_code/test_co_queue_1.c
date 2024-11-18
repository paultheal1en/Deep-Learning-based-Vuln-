static void test_co_queue(void)

{

    Coroutine *c1;

    Coroutine *c2;



    c1 = qemu_coroutine_create(c1_fn);

    c2 = qemu_coroutine_create(c2_fn);



    qemu_coroutine_enter(c1, c2);

    memset(c1, 0xff, sizeof(Coroutine));

    qemu_coroutine_enter(c2, NULL);

}
