static int get_int32(QEMUFile *f, void *pv, size_t size)

{

    int32_t *v = pv;

    qemu_get_sbe32s(f, v);

    return 0;

}
