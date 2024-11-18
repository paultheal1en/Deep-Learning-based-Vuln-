void qemu_aio_unref(void *p)

{

    BlockAIOCB *acb = p;

    assert(acb->refcnt > 0);

    if (--acb->refcnt == 0) {

        g_free(acb);

    }

}
