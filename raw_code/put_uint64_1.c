static void put_uint64(QEMUFile *f, void *pv, size_t size)

{

    uint64_t *v = pv;

    qemu_put_be64s(f, v);

}
