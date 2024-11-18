static void hextile_enc_cord(uint8_t *ptr, int x, int y, int w, int h)

{

    ptr[0] = ((x & 0x0F) << 4) | (y & 0x0F);

    ptr[1] = (((w - 1) & 0x0F) << 4) | ((h - 1) & 0x0F);

}
