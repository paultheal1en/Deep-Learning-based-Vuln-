coroutine_fn iscsi_co_write_zeroes(BlockDriverState *bs, int64_t sector_num,

                                   int nb_sectors, BdrvRequestFlags flags)

{

    IscsiLun *iscsilun = bs->opaque;

    struct IscsiTask iTask;

    uint64_t lba;

    uint32_t nb_blocks;



    if (!is_request_lun_aligned(sector_num, nb_sectors, iscsilun)) {

        return -EINVAL;

    }



    if (!(flags & BDRV_REQ_MAY_UNMAP) && !iscsilun->has_write_same) {

        /* WRITE SAME without UNMAP is not supported by the target */

        return -ENOTSUP;

    }



    if ((flags & BDRV_REQ_MAY_UNMAP) && !iscsilun->lbp.lbpws) {

        /* WRITE SAME with UNMAP is not supported by the target */

        return -ENOTSUP;

    }



    lba = sector_qemu2lun(sector_num, iscsilun);

    nb_blocks = sector_qemu2lun(nb_sectors, iscsilun);



    if (iscsilun->zeroblock == NULL) {

        iscsilun->zeroblock = g_malloc0(iscsilun->block_size);

    }



    iscsi_co_init_iscsitask(iscsilun, &iTask);

retry:

    if (iscsi_writesame16_task(iscsilun->iscsi, iscsilun->lun, lba,

                               iscsilun->zeroblock, iscsilun->block_size,

                               nb_blocks, 0, !!(flags & BDRV_REQ_MAY_UNMAP),

                               0, 0, iscsi_co_generic_cb, &iTask) == NULL) {

        return -ENOMEM;

    }



    while (!iTask.complete) {

        iscsi_set_events(iscsilun);

        qemu_coroutine_yield();

    }



    if (iTask.status == SCSI_STATUS_CHECK_CONDITION &&

        iTask.task->sense.key == SCSI_SENSE_ILLEGAL_REQUEST &&

        iTask.task->sense.ascq == SCSI_SENSE_ASCQ_INVALID_OPERATION_CODE) {

        /* WRITE SAME is not supported by the target */

        iscsilun->has_write_same = false;

        scsi_free_scsi_task(iTask.task);

        return -ENOTSUP;

    }



    if (iTask.task != NULL) {

        scsi_free_scsi_task(iTask.task);

        iTask.task = NULL;

    }



    if (iTask.do_retry) {

        iTask.complete = 0;

        goto retry;

    }



    if (iTask.status != SCSI_STATUS_GOOD) {

        return -EIO;

    }



    return 0;

}
