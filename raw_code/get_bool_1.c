static int get_bool(QEMUFile *f, void *pv, size_t size)

{

    bool *v = pv;

    *v = qemu_get_byte(f);

    return 0;

}
