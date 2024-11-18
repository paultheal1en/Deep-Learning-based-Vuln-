static void qemu_input_event_trace(QemuConsole *src, InputEvent *evt)

{

    const char *name;

    int qcode, idx = -1;



    if (src) {

        idx = qemu_console_get_index(src);

    }

    switch (evt->kind) {

    case INPUT_EVENT_KIND_KEY:

        switch (evt->key->key->kind) {

        case KEY_VALUE_KIND_NUMBER:

            qcode = qemu_input_key_number_to_qcode(evt->key->key->number);

            name = QKeyCode_lookup[qcode];

            trace_input_event_key_number(idx, evt->key->key->number,

                                         name, evt->key->down);

            break;

        case KEY_VALUE_KIND_QCODE:

            name = QKeyCode_lookup[evt->key->key->qcode];

            trace_input_event_key_qcode(idx, name, evt->key->down);

            break;

        case KEY_VALUE_KIND_MAX:

            /* keep gcc happy */

            break;

        }

        break;

    case INPUT_EVENT_KIND_BTN:

        name = InputButton_lookup[evt->btn->button];

        trace_input_event_btn(idx, name, evt->btn->down);

        break;

    case INPUT_EVENT_KIND_REL:

        name = InputAxis_lookup[evt->rel->axis];

        trace_input_event_rel(idx, name, evt->rel->value);

        break;

    case INPUT_EVENT_KIND_ABS:

        name = InputAxis_lookup[evt->abs->axis];

        trace_input_event_abs(idx, name, evt->abs->value);

        break;

    case INPUT_EVENT_KIND_MAX:

        /* keep gcc happy */

        break;

    }

}
