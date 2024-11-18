static void xenfb_mouse_event(DeviceState *dev, QemuConsole *src,

                              InputEvent *evt)

{

    struct XenInput *xenfb = (struct XenInput *)dev;

    InputBtnEvent *btn;

    InputMoveEvent *move;

    QemuConsole *con;

    DisplaySurface *surface;

    int scale;



    switch (evt->type) {

    case INPUT_EVENT_KIND_BTN:

        btn = evt->u.btn.data;

        switch (btn->button) {

        case INPUT_BUTTON_LEFT:

            xenfb_send_key(xenfb, btn->down, BTN_LEFT);

            break;

        case INPUT_BUTTON_RIGHT:

            xenfb_send_key(xenfb, btn->down, BTN_LEFT + 1);

            break;

        case INPUT_BUTTON_MIDDLE:

            xenfb_send_key(xenfb, btn->down, BTN_LEFT + 2);

            break;

        case INPUT_BUTTON_WHEEL_UP:

            if (btn->down) {

                xenfb->wheel--;

            }

            break;

        case INPUT_BUTTON_WHEEL_DOWN:

            if (btn->down) {

                xenfb->wheel++;

            }

            break;

        default:

            break;

        }

        break;



    case INPUT_EVENT_KIND_ABS:

        move = evt->u.abs.data;

        con = qemu_console_lookup_by_index(0);

        if (!con) {

            xen_pv_printf(&xenfb->c.xendev, 0, "No QEMU console available");

            return;

        }

        surface = qemu_console_surface(con);

        switch (move->axis) {

        case INPUT_AXIS_X:

            scale = surface_width(surface) - 1;

            break;

        case INPUT_AXIS_Y:

            scale = surface_height(surface) - 1;

            break;

        default:

            scale = 0x8000;

            break;

        }

        xenfb->axis[move->axis] = move->value * scale / 0x7fff;

        break;



    case INPUT_EVENT_KIND_REL:

        move = evt->u.rel.data;

        xenfb->axis[move->axis] += move->value;

        break;



    default:

        break;

    }

}
