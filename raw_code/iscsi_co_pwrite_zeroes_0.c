coroutine_fn iscsi_co_pwrite_zeroes(BlockDriverState *bs, int64_t offset,

                                    int count, BdrvRequestFlags flags)

{

    IscsiLun *iscsilun = bs->opaque;

    struct IscsiTask iTask;

    uint64_t lba;

    uint32_t nb_blocks;

    bool use_16_for_ws = iscsilun->use_16_for_rw;



    if (!is_byte_request_lun_aligned(offset, count, iscsilun)) {

        return -ENOTSUP;

    }



    if (flags & BDRV_REQ_MAY_UNMAP) {

        if (!use_16_for_ws && !iscsilun->lbp.lbpws10) {

            /* WRITESAME10 with UNMAP is unsupported try WRITESAME16 */

            use_16_for_ws = true;

        }

        if (use_16_for_ws && !iscsilun->lbp.lbpws) {

            /* WRITESAME16 with UNMAP is not supported by the target,

             * fall back and try WRITESAME10/16 without UNMAP */

            flags &= ~BDRV_REQ_MAY_UNMAP;

            use_16_for_ws = iscsilun->use_16_for_rw;

        }

    }



    if (!(flags & BDRV_REQ_MAY_UNMAP) && !iscsilun->has_write_same) {

        /* WRITESAME without UNMAP is not supported by the target */

        return -ENOTSUP;

    }



    lba = offset / iscsilun->block_size;

    nb_blocks = count / iscsilun->block_size;



    if (iscsilun->zeroblock == NULL) {

        iscsilun->zeroblock = g_try_malloc0(iscsilun->block_size);

        if (iscsilun->zeroblock == NULL) {

            return -ENOMEM;

        }

    }



    iscsi_co_init_iscsitask(iscsilun, &iTask);

retry:

    if (use_16_for_ws) {

        iTask.task = iscsi_writesame16_task(iscsilun->iscsi, iscsilun->lun, lba,

                                            iscsilun->zeroblock, iscsilun->block_size,

                                            nb_blocks, 0, !!(flags & BDRV_REQ_MAY_UNMAP),

                                            0, 0, iscsi_co_generic_cb, &iTask);

    } else {

        iTask.task = iscsi_writesame10_task(iscsilun->iscsi, iscsilun->lun, lba,

                                            iscsilun->zeroblock, iscsilun->block_size,

                                            nb_blocks, 0, !!(flags & BDRV_REQ_MAY_UNMAP),

                                            0, 0, iscsi_co_generic_cb, &iTask);

    }

    if (iTask.task == NULL) {

        return -ENOMEM;

    }



    while (!iTask.complete) {

        iscsi_set_events(iscsilun);

        qemu_coroutine_yield();

    }



    if (iTask.status == SCSI_STATUS_CHECK_CONDITION &&

        iTask.task->sense.key == SCSI_SENSE_ILLEGAL_REQUEST &&

        (iTask.task->sense.ascq == SCSI_SENSE_ASCQ_INVALID_OPERATION_CODE ||

         iTask.task->sense.ascq == SCSI_SENSE_ASCQ_INVALID_FIELD_IN_CDB)) {

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

        return iTask.err_code;

    }



    if (flags & BDRV_REQ_MAY_UNMAP) {

        iscsi_allocationmap_clear(iscsilun, offset >> BDRV_SECTOR_BITS,

                                  count >> BDRV_SECTOR_BITS);

    } else {

        iscsi_allocationmap_set(iscsilun, offset >> BDRV_SECTOR_BITS,

                                count >> BDRV_SECTOR_BITS);

    }



    return 0;

}
