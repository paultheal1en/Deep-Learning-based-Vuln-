void replay_input_event(QemuConsole *src, InputEvent *evt)

{

    if (replay_mode == REPLAY_MODE_PLAY) {

        /* Nothing */

    } else if (replay_mode == REPLAY_MODE_RECORD) {

        replay_add_input_event(qapi_clone_InputEvent(evt));

    } else {

        qemu_input_event_send_impl(src, evt);

    }

}
