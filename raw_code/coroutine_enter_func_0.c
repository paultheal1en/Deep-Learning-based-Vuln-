static int coroutine_enter_func(void *arg)

{

    Coroutine *co = arg;

    qemu_coroutine_enter(co, NULL);

    return 0;

}
