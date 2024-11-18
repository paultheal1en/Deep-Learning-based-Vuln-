static void put_int64(QEMUFile *f, void *pv, size_t size)

{

    int64_t *v = pv;

    qemu_put_sbe64s(f, v);

}
