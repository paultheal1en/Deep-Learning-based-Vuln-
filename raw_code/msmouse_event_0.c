static void msmouse_event(void *opaque,

                          int dx, int dy, int dz, int buttons_state)

{

    CharDriverState *chr = (CharDriverState *)opaque;



    unsigned char bytes[4] = { 0x40, 0x00, 0x00, 0x00 };



    /* Movement deltas */

    bytes[0] |= (MSMOUSE_HI2(dy) << 2) | MSMOUSE_HI2(dx);

    bytes[1] |= MSMOUSE_LO6(dx);

    bytes[2] |= MSMOUSE_LO6(dy);



    /* Buttons */

    bytes[0] |= (buttons_state & 0x01 ? 0x20 : 0x00);

    bytes[0] |= (buttons_state & 0x02 ? 0x10 : 0x00);

    bytes[3] |= (buttons_state & 0x04 ? 0x20 : 0x00);



    /* We always send the packet of, so that we do not have to keep track

       of previous state of the middle button. This can potentially confuse

       some very old drivers for two button mice though. */

    qemu_chr_be_write(chr, bytes, 4);

}
