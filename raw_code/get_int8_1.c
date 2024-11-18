static int get_int8(QEMUFile *f, void *pv, size_t size)

{

    int8_t *v = pv;

    qemu_get_s8s(f, v);

    return 0;

}
