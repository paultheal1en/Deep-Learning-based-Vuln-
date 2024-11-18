static void coroutine_fn c1_fn(void *opaque)

{

    Coroutine *c2 = opaque;

    qemu_coroutine_enter(c2, NULL);

}
