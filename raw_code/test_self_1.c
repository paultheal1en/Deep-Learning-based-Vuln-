static void test_self(void)

{

    Coroutine *coroutine;



    coroutine = qemu_coroutine_create(verify_self);

    qemu_coroutine_enter(coroutine, &coroutine);

}
