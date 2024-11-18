iscsi_allocmap_update(IscsiLun *iscsilun, int64_t sector_num,

                      int nb_sectors, bool allocated, bool valid)

{

    int64_t cl_num_expanded, nb_cls_expanded, cl_num_shrunk, nb_cls_shrunk;



    if (iscsilun->allocmap == NULL) {

        return;

    }

    /* expand to entirely contain all affected clusters */

    cl_num_expanded = sector_num / iscsilun->cluster_sectors;

    nb_cls_expanded = DIV_ROUND_UP(sector_num + nb_sectors,

                                   iscsilun->cluster_sectors) - cl_num_expanded;

    /* shrink to touch only completely contained clusters */

    cl_num_shrunk = DIV_ROUND_UP(sector_num, iscsilun->cluster_sectors);

    nb_cls_shrunk = (sector_num + nb_sectors) / iscsilun->cluster_sectors

                      - cl_num_shrunk;

    if (allocated) {

        bitmap_set(iscsilun->allocmap, cl_num_expanded, nb_cls_expanded);

    } else {

        bitmap_clear(iscsilun->allocmap, cl_num_shrunk, nb_cls_shrunk);

    }



    if (iscsilun->allocmap_valid == NULL) {

        return;

    }

    if (valid) {

        bitmap_set(iscsilun->allocmap_valid, cl_num_shrunk, nb_cls_shrunk);

    } else {

        bitmap_clear(iscsilun->allocmap_valid, cl_num_expanded,

                     nb_cls_expanded);

    }

}
