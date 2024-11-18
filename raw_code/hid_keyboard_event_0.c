static void hid_keyboard_event(DeviceState *dev, QemuConsole *src,

                               InputEvent *evt)

{

    HIDState *hs = (HIDState *)dev;

    int scancodes[3], i, count;

    int slot;



    count = qemu_input_key_value_to_scancode(evt->key->key,

                                             evt->key->down,

                                             scancodes);

    if (hs->n + count > QUEUE_LENGTH) {

        fprintf(stderr, "usb-kbd: warning: key event queue full\n");

        return;

    }

    for (i = 0; i < count; i++) {

        slot = (hs->head + hs->n) & QUEUE_MASK; hs->n++;

        hs->kbd.keycodes[slot] = scancodes[i];

    }

    hs->event(hs);

}
