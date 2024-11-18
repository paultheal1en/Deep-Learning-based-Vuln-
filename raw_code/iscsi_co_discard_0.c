coroutine_fn iscsi_co_discard(BlockDriverState *bs, int64_t sector_num,

                                   int nb_sectors)

{

    IscsiLun *iscsilun = bs->opaque;

    struct IscsiTask iTask;

    struct unmap_list list;

    uint32_t nb_blocks;

    uint32_t max_unmap;



    if (!is_request_lun_aligned(sector_num, nb_sectors, iscsilun)) {

        return -EINVAL;

    }



    if (!iscsilun->lbp.lbpu) {

        /* UNMAP is not supported by the target */

        return 0;

    }



    list.lba = sector_qemu2lun(sector_num, iscsilun);

    nb_blocks = sector_qemu2lun(nb_sectors, iscsilun);



    max_unmap = iscsilun->bl.max_unmap;

    if (max_unmap == 0xffffffff) {

        max_unmap = ISCSI_MAX_UNMAP;

    }



    while (nb_blocks > 0) {

        iscsi_co_init_iscsitask(iscsilun, &iTask);

        list.num = nb_blocks;

        if (list.num > max_unmap) {

            list.num = max_unmap;

        }

retry:

        if (iscsi_unmap_task(iscsilun->iscsi, iscsilun->lun, 0, 0, &list, 1,

                         iscsi_co_generic_cb, &iTask) == NULL) {

            return -EIO;

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

            goto retry;

        }



        if (iTask.status == SCSI_STATUS_CHECK_CONDITION) {

            /* the target might fail with a check condition if it

               is not happy with the alignment of the UNMAP request

               we silently fail in this case */

            return 0;

        }



        if (iTask.status != SCSI_STATUS_GOOD) {

            return -EIO;

        }



        list.lba += list.num;

        nb_blocks -= list.num;

    }



    return 0;

}
