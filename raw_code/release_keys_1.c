static void release_keys(void *opaque)

{

    int i;



    for (i = 0; i < keycodes_size; i++) {

        if (keycodes[i] & 0x80) {

            kbd_put_keycode(0xe0);

        }

        kbd_put_keycode(keycodes[i]| 0x80);

    }



    free_keycodes();

}
