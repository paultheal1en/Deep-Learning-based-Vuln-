void qemu_chr_fe_printf(CharDriverState *s, const char *fmt, ...)

{

    char buf[READ_BUF_LEN];

    va_list ap;

    va_start(ap, fmt);

    vsnprintf(buf, sizeof(buf), fmt, ap);

    qemu_chr_fe_write(s, (uint8_t *)buf, strlen(buf));

    va_end(ap);

}
