void kbd_put_keycode(int keycode)

{

    QEMUPutKbdEntry *entry = QTAILQ_FIRST(&kbd_handlers);



    if (!runstate_is_running() && !runstate_check(RUN_STATE_SUSPENDED)) {

        return;

    }

    if (entry) {

        entry->put_kbd(entry->opaque, keycode);

    }

}
