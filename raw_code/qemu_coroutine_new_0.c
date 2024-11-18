Coroutine *qemu_coroutine_new(void)

{

    CoroutineThreadState *s = coroutine_get_thread_state();

    Coroutine *co;



    co = QLIST_FIRST(&s->pool);

    if (co) {

        QLIST_REMOVE(co, pool_next);

        s->pool_size--;

    } else {

        co = coroutine_new();

    }

    return co;

}
