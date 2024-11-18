static int get_float64(QEMUFile *f, void *pv, size_t size)

{

    float64 *v = pv;



    *v = make_float64(qemu_get_be64(f));

    return 0;

}
