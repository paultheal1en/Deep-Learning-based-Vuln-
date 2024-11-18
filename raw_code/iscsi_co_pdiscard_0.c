coroutine_fn iscsi_co_pdiscard(BlockDriverState *bs, int64_t offset, int count)

{

    IscsiLun *iscsilun = bs->opaque;

    struct IscsiTask iTask;

    struct unmap_list list;



    if (!is_byte_request_lun_aligned(offset, count, iscsilun)) {

        return -ENOTSUP;

    }



    if (!iscsilun->lbp.lbpu) {

        /* UNMAP is not supported by the target */

        return 0;

    }



    list.lba = offset / iscsilun->block_size;

    list.num = count / iscsilun->block_size;



    iscsi_co_init_iscsitask(iscsilun, &iTask);

retry:

    if (iscsi_unmap_task(iscsilun->iscsi, iscsilun->lun, 0, 0, &list, 1,

                         iscsi_co_generic_cb, &iTask) == NULL) {

        return -ENOMEM;

    }



    while (!iTask.complete) {

        iscsi_set_events(iscsilun);

        qemu_coroutine_yield();

    }



    if (iTask.task != NULL) {

        scsi_free_scsi_task(iTask.task);

        iTask.task = NULL;

    }



    if (iTask.do_retry) {

        iTask.complete = 0;

        goto retry;

    }



    if (iTask.status == SCSI_STATUS_CHECK_CONDITION) {

        /* the target might fail with a check condition if it

           is not happy with the alignment of the UNMAP request

           we silently fail in this case */

        return 0;

    }



    if (iTask.status != SCSI_STATUS_GOOD) {

        return iTask.err_code;

    }



    iscsi_allocmap_set_invalid(iscsilun, offset >> BDRV_SECTOR_BITS,

                               count >> BDRV_SECTOR_BITS);



    return 0;

}
