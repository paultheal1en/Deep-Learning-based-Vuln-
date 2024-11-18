static void put_int8(QEMUFile *f, void *pv, size_t size)

{

    int8_t *v = pv;

    qemu_put_s8s(f, v);

}
