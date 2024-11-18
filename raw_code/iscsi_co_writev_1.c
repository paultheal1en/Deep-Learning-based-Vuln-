static int coroutine_fn iscsi_co_writev(BlockDriverState *bs,

                                        int64_t sector_num, int nb_sectors,

                                        QEMUIOVector *iov)

{

    IscsiLun *iscsilun = bs->opaque;

    struct IscsiTask iTask;

    uint64_t lba;

    uint32_t num_sectors;

    uint8_t *data = NULL;

    uint8_t *buf = NULL;



    if (!is_request_lun_aligned(sector_num, nb_sectors, iscsilun)) {

        return -EINVAL;

    }



    lba = sector_qemu2lun(sector_num, iscsilun);

    num_sectors = sector_qemu2lun(nb_sectors, iscsilun);

#if !defined(LIBISCSI_FEATURE_IOVECTOR)

    /* if the iovec only contains one buffer we can pass it directly */

    if (iov->niov == 1) {

        data = iov->iov[0].iov_base;

    } else {

        size_t size = MIN(nb_sectors * BDRV_SECTOR_SIZE, iov->size);

        buf = g_malloc(size);

        qemu_iovec_to_buf(iov, 0, buf, size);

        data = buf;

    }

#endif

    iscsi_co_init_iscsitask(iscsilun, &iTask);

retry:

    if (iscsilun->use_16_for_rw) {

        iTask.task = iscsi_write16_task(iscsilun->iscsi, iscsilun->lun, lba,

                                        data, num_sectors * iscsilun->block_size,

                                        iscsilun->block_size, 0, 0, 0, 0, 0,

                                        iscsi_co_generic_cb, &iTask);

    } else {

        iTask.task = iscsi_write10_task(iscsilun->iscsi, iscsilun->lun, lba,

                                        data, num_sectors * iscsilun->block_size,

                                        iscsilun->block_size, 0, 0, 0, 0, 0,

                                        iscsi_co_generic_cb, &iTask);

    }

    if (iTask.task == NULL) {

        g_free(buf);

        return -ENOMEM;

    }

#if defined(LIBISCSI_FEATURE_IOVECTOR)

    scsi_task_set_iov_out(iTask.task, (struct scsi_iovec *) iov->iov,

                          iov->niov);

#endif

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



    g_free(buf);



    if (iTask.status != SCSI_STATUS_GOOD) {

        return -EIO;

    }



    iscsi_allocationmap_set(iscsilun, sector_num, nb_sectors);



    return 0;

}
