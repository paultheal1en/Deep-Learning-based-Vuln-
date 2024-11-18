InputEvent *qemu_input_event_new_key(KeyValue *key, bool down)

{

    InputEvent *evt = g_new0(InputEvent, 1);

    evt->key = g_new0(InputKeyEvent, 1);

    evt->kind = INPUT_EVENT_KIND_KEY;

    evt->key->key = key;

    evt->key->down = down;

    return evt;

}
