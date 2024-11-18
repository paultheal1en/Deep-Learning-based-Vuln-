static void vnc_write_u16(VncState *vs, uint16_t value)

{

    uint8_t buf[2];



    buf[0] = (value >> 8) & 0xFF;

    buf[1] = value & 0xFF;



    vnc_write(vs, buf, 2);

}
