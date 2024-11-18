void qemu_sglist_destroy(QEMUSGList *qsg)

{

    g_free(qsg->sg);


}