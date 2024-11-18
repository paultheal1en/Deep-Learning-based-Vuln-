int qemu_chr_fe_write(CharDriverState *s, const uint8_t *buf, int len)

{

    return s->chr_write(s, buf, len);

}
