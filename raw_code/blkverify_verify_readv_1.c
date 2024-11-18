static void blkverify_verify_readv(BlkverifyAIOCB *acb)

{

    ssize_t offset = qemu_iovec_compare(acb->qiov, &acb->raw_qiov);

    if (offset != -1) {

        blkverify_err(acb, "contents mismatch in sector %" PRId64,

                      acb->sector_num + (int64_t)(offset / BDRV_SECTOR_SIZE));

    }

}
