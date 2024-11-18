static void tm_get(QEMUFile *f, struct tm *tm) {

    tm->tm_sec = qemu_get_be16(f);

    tm->tm_min = qemu_get_be16(f);

    tm->tm_hour = qemu_get_be16(f);

    tm->tm_mday = qemu_get_be16(f);

    tm->tm_min = qemu_get_be16(f);

    tm->tm_year = qemu_get_be16(f);

}
