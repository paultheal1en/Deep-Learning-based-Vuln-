static void put_buffer(QEMUFile *f, void *pv, size_t size)

{

    uint8_t *v = pv;

    qemu_put_buffer(f, v, size);

}
