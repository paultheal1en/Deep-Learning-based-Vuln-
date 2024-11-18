static void tight_send_compact_size(VncState *vs, size_t len)

{

    int lpc = 0;

    int bytes = 0;

    char buf[3] = {0, 0, 0};



    buf[bytes++] = len & 0x7F;

    if (len > 0x7F) {

        buf[bytes-1] |= 0x80;

        buf[bytes++] = (len >> 7) & 0x7F;

        if (len > 0x3FFF) {

            buf[bytes-1] |= 0x80;

            buf[bytes++] = (len >> 14) & 0xFF;

        }

    }

    for (lpc = 0; lpc < bytes; lpc++) {

        vnc_write_u8(vs, buf[lpc]);

    }

}
