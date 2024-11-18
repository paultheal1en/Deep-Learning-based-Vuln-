void qemu_put_be16(QEMUFile *f, unsigned int v)

{

    qemu_put_byte(f, v >> 8);

    qemu_put_byte(f, v);

}
