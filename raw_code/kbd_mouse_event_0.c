void kbd_mouse_event(int dx, int dy, int dz, int buttons_state)

{

    QEMUPutMouseEntry *entry;

    QEMUPutMouseEvent *mouse_event;

    void *mouse_event_opaque;

    int width, height;



    if (!runstate_is_running()) {

        return;

    }

    if (QTAILQ_EMPTY(&mouse_handlers)) {

        return;

    }



    entry = QTAILQ_FIRST(&mouse_handlers);



    mouse_event = entry->qemu_put_mouse_event;

    mouse_event_opaque = entry->qemu_put_mouse_event_opaque;



    if (mouse_event) {

        if (entry->qemu_put_mouse_event_absolute) {

            width = 0x7fff;

            height = 0x7fff;

        } else {

            width = graphic_width - 1;

            height = graphic_height - 1;

        }



        switch (graphic_rotate) {

        case 0:

            mouse_event(mouse_event_opaque,

                        dx, dy, dz, buttons_state);

            break;

        case 90:

            mouse_event(mouse_event_opaque,

                        width - dy, dx, dz, buttons_state);

            break;

        case 180:

            mouse_event(mouse_event_opaque,

                        width - dx, height - dy, dz, buttons_state);

            break;

        case 270:

            mouse_event(mouse_event_opaque,

                        dy, height - dx, dz, buttons_state);

            break;

        }

    }

}
