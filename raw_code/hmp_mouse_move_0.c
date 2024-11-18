static void hmp_mouse_move(Monitor *mon, const QDict *qdict)

{

    int dx, dy, dz, button;

    const char *dx_str = qdict_get_str(qdict, "dx_str");

    const char *dy_str = qdict_get_str(qdict, "dy_str");

    const char *dz_str = qdict_get_try_str(qdict, "dz_str");



    dx = strtol(dx_str, NULL, 0);

    dy = strtol(dy_str, NULL, 0);

    qemu_input_queue_rel(NULL, INPUT_AXIS_X, dx);

    qemu_input_queue_rel(NULL, INPUT_AXIS_Y, dy);



    if (dz_str) {

        dz = strtol(dz_str, NULL, 0);

        if (dz != 0) {

            button = (dz > 0) ? INPUT_BUTTON_WHEEL_UP : INPUT_BUTTON_WHEEL_DOWN;

            qemu_input_queue_btn(NULL, button, true);

            qemu_input_event_sync();

            qemu_input_queue_btn(NULL, button, false);

        }

    }

    qemu_input_event_sync();

}
