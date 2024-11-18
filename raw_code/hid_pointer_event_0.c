static void hid_pointer_event(DeviceState *dev, QemuConsole *src,

                              InputEvent *evt)

{

    static const int bmap[INPUT_BUTTON__MAX] = {

        [INPUT_BUTTON_LEFT]   = 0x01,

        [INPUT_BUTTON_RIGHT]  = 0x02,

        [INPUT_BUTTON_MIDDLE] = 0x04,

    };

    HIDState *hs = (HIDState *)dev;

    HIDPointerEvent *e;

    InputMoveEvent *move;

    InputBtnEvent *btn;



    assert(hs->n < QUEUE_LENGTH);

    e = &hs->ptr.queue[(hs->head + hs->n) & QUEUE_MASK];



    switch (evt->type) {

    case INPUT_EVENT_KIND_REL:

        move = evt->u.rel;

        if (move->axis == INPUT_AXIS_X) {

            e->xdx += move->value;

        } else if (move->axis == INPUT_AXIS_Y) {

            e->ydy += move->value;

        }

        break;



    case INPUT_EVENT_KIND_ABS:

        move = evt->u.abs;

        if (move->axis == INPUT_AXIS_X) {

            e->xdx = move->value;

        } else if (move->axis == INPUT_AXIS_Y) {

            e->ydy = move->value;

        }

        break;



    case INPUT_EVENT_KIND_BTN:

        btn = evt->u.btn;

        if (btn->down) {

            e->buttons_state |= bmap[btn->button];

            if (btn->button == INPUT_BUTTON_WHEEL_UP) {

                e->dz--;

            } else if (btn->button == INPUT_BUTTON_WHEEL_DOWN) {

                e->dz++;

            }

        } else {

            e->buttons_state &= ~bmap[btn->button];

        }

        break;



    default:

        /* keep gcc happy */

        break;

    }



}
