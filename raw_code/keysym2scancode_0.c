int keysym2scancode(void *kbd_layout, int keysym)

{

    kbd_layout_t *k = kbd_layout;

    if (keysym < MAX_NORMAL_KEYCODE) {

        if (k->keysym2keycode[keysym] == 0) {

            trace_keymap_unmapped(keysym);

            fprintf(stderr, "Warning: no scancode found for keysym %d\n",

                    keysym);

        }

        return k->keysym2keycode[keysym];

    } else {

        int i;

#ifdef XK_ISO_Left_Tab

        if (keysym == XK_ISO_Left_Tab) {

            keysym = XK_Tab;

        }

#endif

        for (i = 0; i < k->extra_count; i++) {

            if (k->keysym2keycode_extra[i].keysym == keysym) {

                return k->keysym2keycode_extra[i].keycode;

            }

        }

    }

    return 0;

}
