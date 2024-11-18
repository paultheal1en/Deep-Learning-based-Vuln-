void qemu_input_event_sync(void)

{

    QemuInputHandlerState *s;



    if (!runstate_is_running() && !runstate_check(RUN_STATE_SUSPENDED)) {

        return;

    }



    trace_input_event_sync();



    QTAILQ_FOREACH(s, &handlers, node) {

        if (!s->events) {

            continue;

        }

        if (s->handler->sync) {

            s->handler->sync(s->dev);

        }

        s->events = 0;

    }

}
