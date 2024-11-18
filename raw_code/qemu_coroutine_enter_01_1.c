void qemu_coroutine_enter(Coroutine *co, void *opaque)

{

    Coroutine *self = qemu_coroutine_self();

    CoroutineAction ret;



    trace_qemu_coroutine_enter(self, co, opaque);



    if (co->caller) {

        fprintf(stderr, "Co-routine re-entered recursively\n");

        abort();

    }



    co->caller = self;

    co->entry_arg = opaque;

    ret = qemu_coroutine_switch(self, co, COROUTINE_ENTER);



    qemu_co_queue_run_restart(co);



    switch (ret) {

    case COROUTINE_YIELD:

        return;

    case COROUTINE_TERMINATE:

        trace_qemu_coroutine_terminate(co);

        coroutine_delete(co);

        return;

    default:

        abort();

    }

}
