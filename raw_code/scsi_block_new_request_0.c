static SCSIRequest *scsi_block_new_request(SCSIDevice *d, uint32_t tag,

                                           uint32_t lun, uint8_t *buf,

                                           void *hba_private)

{

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, d);



    switch (buf[0]) {

    case READ_6:

    case READ_10:

    case READ_12:

    case READ_16:

    case VERIFY_10:

    case VERIFY_12:

    case VERIFY_16:

    case WRITE_6:

    case WRITE_10:

    case WRITE_12:

    case WRITE_16:

    case WRITE_VERIFY_10:

    case WRITE_VERIFY_12:

    case WRITE_VERIFY_16:

        /* If we are not using O_DIRECT, we might read stale data from the

	 * host cache if writes were made using other commands than these

	 * ones (such as WRITE SAME or EXTENDED COPY, etc.).  So, without

	 * O_DIRECT everything must go through SG_IO.

         */

        if (bdrv_get_flags(s->qdev.conf.bs) & BDRV_O_NOCACHE) {

            break;

        }



        /* MMC writing cannot be done via pread/pwrite, because it sometimes

         * involves writing beyond the maximum LBA or to negative LBA (lead-in).

         * And once you do these writes, reading from the block device is

         * unreliable, too.  It is even possible that reads deliver random data

         * from the host page cache (this is probably a Linux bug).

         *

         * We might use scsi_disk_reqops as long as no writing commands are

         * seen, but performance usually isn't paramount on optical media.  So,

         * just make scsi-block operate the same as scsi-generic for them.

         */

        if (s->qdev.type == TYPE_ROM) {

            break;

	}

        return scsi_req_alloc(&scsi_disk_reqops, &s->qdev, tag, lun,

                              hba_private);

    }



    return scsi_req_alloc(&scsi_generic_req_ops, &s->qdev, tag, lun,

                          hba_private);

}
