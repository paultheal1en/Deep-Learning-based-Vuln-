const QEMUSizedBuffer *qemu_buf_get(QEMUFile *f)

{

    QEMUBuffer *p;



    qemu_fflush(f);



    p = f->opaque;



    return p->qsb;

}
