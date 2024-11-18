static void put_int32(QEMUFile *f, void *pv, size_t size)

{

    int32_t *v = pv;

    qemu_put_sbe32s(f, v);

}
