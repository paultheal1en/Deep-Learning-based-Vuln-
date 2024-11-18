static void restart_co_req(void *opaque)

{

    Coroutine *co = opaque;



    qemu_coroutine_enter(co, NULL);

}
