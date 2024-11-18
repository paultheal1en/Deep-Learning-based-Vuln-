int get_osversion(void)

{

    static int osversion;

    struct new_utsname buf;

    const char *s;

    int i, n, tmp;

    if (osversion)

        return osversion;

    if (qemu_uname_release && *qemu_uname_release) {

        s = qemu_uname_release;

    } else {

        if (sys_uname(&buf))

            return 0;

        s = buf.release;

    }

    tmp = 0;

    for (i = 0; i < 3; i++) {

        n = 0;

        while (*s >= '0' && *s <= '9') {

            n *= 10;

            n += *s - '0';

            s++;

        }

        tmp = (tmp << 8) + n;

        if (*s == '.')

            s++;

    }

    osversion = tmp;

    return osversion;

}
