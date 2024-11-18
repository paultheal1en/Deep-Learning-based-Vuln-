static void put_uint16(QEMUFile *f, void *pv, size_t size)

{

    uint16_t *v = pv;

    qemu_put_be16s(f, v);

}
