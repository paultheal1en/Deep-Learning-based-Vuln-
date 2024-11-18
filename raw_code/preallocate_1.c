static int preallocate(BlockDriverState *bs)

{

    uint64_t nb_sectors;

    uint64_t offset;

    int num;

    int ret;

    QCowL2Meta meta;



    nb_sectors = bdrv_getlength(bs) >> 9;

    offset = 0;

    QLIST_INIT(&meta.dependent_requests);

    meta.cluster_offset = 0;



    while (nb_sectors) {

        num = MIN(nb_sectors, INT_MAX >> 9);

        ret = qcow2_alloc_cluster_offset(bs, offset, 0, num, &num, &meta);



        if (ret < 0) {

            return -1;

        }



        if (qcow2_alloc_cluster_link_l2(bs, &meta) < 0) {

            qcow2_free_any_clusters(bs, meta.cluster_offset, meta.nb_clusters);

            return -1;

        }



        /* There are no dependent requests, but we need to remove our request

         * from the list of in-flight requests */

        run_dependent_requests(&meta);



        /* TODO Preallocate data if requested */



        nb_sectors -= num;

        offset += num << 9;

    }



    /*

     * It is expected that the image file is large enough to actually contain

     * all of the allocated clusters (otherwise we get failing reads after

     * EOF). Extend the image to the last allocated sector.

     */

    if (meta.cluster_offset != 0) {

        uint8_t buf[512];

        memset(buf, 0, 512);

        bdrv_write(bs->file, (meta.cluster_offset >> 9) + num - 1, buf, 1);

    }



    return 0;

}
