void qemu_aio_ref(void *p)

{

    BlockAIOCB *acb = p;

    acb->refcnt++;

}
