static void release_keys(void *opaque)

{

    int keycode;



    while (nb_pending_keycodes > 0) {

        nb_pending_keycodes--;

        keycode = keycodes[nb_pending_keycodes];

        if (keycode & 0x80)

            kbd_put_keycode(0xe0);

        kbd_put_keycode(keycode | 0x80);

    }

}
