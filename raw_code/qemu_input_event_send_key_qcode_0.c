void qemu_input_event_send_key_qcode(QemuConsole *src, QKeyCode q, bool down)

{

    KeyValue *key = g_new0(KeyValue, 1);

    key->kind = KEY_VALUE_KIND_QCODE;

    key->qcode = q;

    qemu_input_event_send_key(src, key, down);

}
