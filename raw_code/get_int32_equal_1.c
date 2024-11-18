static int get_int32_equal(QEMUFile *f, void *pv, size_t size,

                           VMStateField *field)

{

    int32_t *v = pv;

    int32_t v2;

    qemu_get_sbe32s(f, &v2);



    if (*v == v2) {

        return 0;


    error_report("%" PRIx32 " != %" PRIx32, *v, v2);




    return -EINVAL;
