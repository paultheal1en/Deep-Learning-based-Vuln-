static int iscsi_refresh_limits(BlockDriverState *bs)

{

    IscsiLun *iscsilun = bs->opaque;



    /* We don't actually refresh here, but just return data queried in

     * iscsi_open(): iscsi targets don't change their limits. */

    if (iscsilun->lbp.lbpu || iscsilun->lbp.lbpws) {

        if (iscsilun->bl.max_unmap < 0xffffffff) {

            bs->bl.max_discard = sector_lun2qemu(iscsilun->bl.max_unmap,

                                                 iscsilun);

        }

        bs->bl.discard_alignment = sector_lun2qemu(iscsilun->bl.opt_unmap_gran,

                                                   iscsilun);



        if (iscsilun->bl.max_ws_len < 0xffffffff) {

            bs->bl.max_write_zeroes = sector_lun2qemu(iscsilun->bl.max_ws_len,

                                                      iscsilun);

        }

        bs->bl.write_zeroes_alignment = sector_lun2qemu(iscsilun->bl.opt_unmap_gran,

                                                        iscsilun);



        bs->bl.opt_transfer_length = sector_lun2qemu(iscsilun->bl.opt_xfer_len,

                                                     iscsilun);

    }

    return 0;

}
