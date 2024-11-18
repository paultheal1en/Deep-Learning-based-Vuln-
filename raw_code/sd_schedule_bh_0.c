static int sd_schedule_bh(QEMUBHFunc *cb, SheepdogAIOCB *acb)

{

    if (acb->bh) {

        error_report("bug: %d %d\n", acb->aiocb_type, acb->aiocb_type);

        return -EIO;

    }



    acb->bh = qemu_bh_new(cb, acb);

    if (!acb->bh) {

        error_report("oom: %d %d\n", acb->aiocb_type, acb->aiocb_type);

        return -EIO;

    }



    qemu_bh_schedule(acb->bh);



    return 0;

}
