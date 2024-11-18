void monitor_vprintf(Monitor *mon, const char *fmt, va_list ap)

{

    if (!mon)

        return;



    if (mon->mc && !mon->mc->print_enabled) {

        qemu_error_new(QERR_UNDEFINED_ERROR);

    } else {

        char buf[4096];

        vsnprintf(buf, sizeof(buf), fmt, ap);

        monitor_puts(mon, buf);

    }

}
