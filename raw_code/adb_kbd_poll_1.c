static int adb_kbd_poll(ADBDevice *d, uint8_t *obuf)

{

    KBDState *s = ADB_KEYBOARD(d);

    int keycode;

    int olen;



    olen = 0;

    if (s->count == 0) {

        return 0;

    }

    keycode = s->data[s->rptr];

    s->rptr++;

    if (s->rptr == sizeof(s->data)) {

        s->rptr = 0;

    }

    s->count--;

    /*

     * The power key is the only two byte value key, so it is a special case.

     * Since 0x7f is not a used keycode for ADB we overload it to indicate the

     * power button when we're storing keycodes in our internal buffer, and

     * expand it out to two bytes when we send to the guest.

     */

    if (keycode == 0x7f) {

        obuf[0] = 0x7f;

        obuf[1] = 0x7f;

        olen = 2;

    } else {

        obuf[0] = keycode;

        /* NOTE: the power key key-up is the two byte sequence 0xff 0xff;

         * otherwise we could in theory send a second keycode in the second

         * byte, but choose not to bother.

         */

        obuf[1] = 0xff;

        olen = 2;

    }



    return olen;

}
