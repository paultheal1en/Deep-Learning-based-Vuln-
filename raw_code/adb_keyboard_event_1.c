static void adb_keyboard_event(DeviceState *dev, QemuConsole *src,

                               InputEvent *evt)

{

    KBDState *s = (KBDState *)dev;

    int qcode, keycode;



    qcode = qemu_input_key_value_to_qcode(evt->u.key.data->key);

    if (qcode >= ARRAY_SIZE(qcode_to_adb_keycode)) {

        return;

    }

    /* FIXME: take handler into account when translating qcode */

    keycode = qcode_to_adb_keycode[qcode];

    if (keycode == NO_KEY) {  /* We don't want to send this to the guest */

        ADB_DPRINTF("Ignoring NO_KEY\n");

        return;

    }

    if (evt->u.key.data->down == false) { /* if key release event */

        keycode = keycode | 0x80;   /* create keyboard break code */

    }



    adb_kbd_put_keycode(s, keycode);

}
