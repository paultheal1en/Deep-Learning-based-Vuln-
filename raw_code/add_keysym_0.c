static void add_keysym(char *line, int keysym, int keycode, kbd_layout_t *k) {

    if (keysym < MAX_NORMAL_KEYCODE) {

        trace_keymap_add("normal", keysym, keycode, line);

        k->keysym2keycode[keysym] = keycode;

    } else {

        if (k->extra_count >= MAX_EXTRA_COUNT) {

            fprintf(stderr, "Warning: Could not assign keysym %s (0x%x)"

                    " because of memory constraints.\n", line, keysym);

        } else {

            trace_keymap_add("extra", keysym, keycode, line);

            k->keysym2keycode_extra[k->extra_count].

            keysym = keysym;

            k->keysym2keycode_extra[k->extra_count].

            keycode = keycode;

            k->extra_count++;

        }

    }

}
