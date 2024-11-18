InputEvent *qemu_input_event_new_btn(InputButton btn, bool down)

{

    InputEvent *evt = g_new0(InputEvent, 1);

    evt->btn = g_new0(InputBtnEvent, 1);

    evt->kind = INPUT_EVENT_KIND_BTN;

    evt->btn->button = btn;

    evt->btn->down = down;

    return evt;

}
