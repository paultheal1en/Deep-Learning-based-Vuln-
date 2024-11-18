void term_printf(const char *fmt, ...)

{

    char buf[4096];

    va_list ap;

    va_start(ap, fmt);

    vsnprintf(buf, sizeof(buf), fmt, ap);

    qemu_chr_write(monitor_hd, buf, strlen(buf));

    va_end(ap);

}
