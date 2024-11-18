static unsigned long *iscsi_allocationmap_init(IscsiLun *iscsilun)

{

    return bitmap_try_new(DIV_ROUND_UP(sector_lun2qemu(iscsilun->num_blocks,

                                                       iscsilun),

                                       iscsilun->cluster_sectors));

}
