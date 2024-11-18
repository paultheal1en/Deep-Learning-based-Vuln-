static void put_float64(QEMUFile *f, void *pv, size_t size)

{

    uint64_t *v = pv;



    qemu_put_be64(f, float64_val(*v));

}
