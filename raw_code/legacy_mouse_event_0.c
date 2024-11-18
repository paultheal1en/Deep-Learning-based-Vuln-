static void legacy_mouse_event(DeviceState *dev, QemuConsole *src,

                               InputEvent *evt)

{

    static const int bmap[INPUT_BUTTON__MAX] = {

        [INPUT_BUTTON_LEFT]   = MOUSE_EVENT_LBUTTON,

        [INPUT_BUTTON_MIDDLE] = MOUSE_EVENT_MBUTTON,

        [INPUT_BUTTON_RIGHT]  = MOUSE_EVENT_RBUTTON,

    };

    QEMUPutMouseEntry *s = (QEMUPutMouseEntry *)dev;

    InputBtnEvent *btn;

    InputMoveEvent *move;



    switch (evt->type) {

    case INPUT_EVENT_KIND_BTN:

        btn = evt->u.btn;

        if (btn->down) {

            s->buttons |= bmap[btn->button];

        } else {

            s->buttons &= ~bmap[btn->button];

        }

        if (btn->down && btn->button == INPUT_BUTTON_WHEEL_UP) {

            s->qemu_put_mouse_event(s->qemu_put_mouse_event_opaque,

                                    s->axis[INPUT_AXIS_X],

                                    s->axis[INPUT_AXIS_Y],

                                    -1,

                                    s->buttons);

        }

        if (btn->down && btn->button == INPUT_BUTTON_WHEEL_DOWN) {

            s->qemu_put_mouse_event(s->qemu_put_mouse_event_opaque,

                                    s->axis[INPUT_AXIS_X],

                                    s->axis[INPUT_AXIS_Y],

                                    1,

                                    s->buttons);

        }

        break;

    case INPUT_EVENT_KIND_ABS:

        move = evt->u.abs;

        s->axis[move->axis] = move->value;

        break;

    case INPUT_EVENT_KIND_REL:

        move = evt->u.rel;

        s->axis[move->axis] += move->value;

        break;

    default:

        break;

    }

}
