void qemu_put_be64(QEMUFile *f, uint64_t v)

{

    qemu_put_be32(f, v >> 32);

    qemu_put_be32(f, v);

}
