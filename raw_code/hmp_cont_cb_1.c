static void hmp_cont_cb(void *opaque, int err)

{

    if (!err) {

        qmp_cont(NULL);

    }

}
