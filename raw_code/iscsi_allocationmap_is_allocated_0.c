static bool iscsi_allocationmap_is_allocated(IscsiLun *iscsilun,

                                             int64_t sector_num, int nb_sectors)

{

    unsigned long size;

    if (iscsilun->allocationmap == NULL) {

        return true;

    }

    size = DIV_ROUND_UP(sector_num + nb_sectors, iscsilun->cluster_sectors);

    return !(find_next_bit(iscsilun->allocationmap, size,

                           sector_num / iscsilun->cluster_sectors) == size);

}
