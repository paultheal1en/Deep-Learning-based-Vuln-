static void press_key(VncState *vs, int keysym)

{

    int keycode = keysym2scancode(vs->vd->kbd_layout, keysym) & SCANCODE_KEYMASK;

    qemu_input_event_send_key_number(vs->vd->dcl.con, keycode, true);

    qemu_input_event_send_key_delay(0);

    qemu_input_event_send_key_number(vs->vd->dcl.con, keycode, false);

    qemu_input_event_send_key_delay(0);

}
