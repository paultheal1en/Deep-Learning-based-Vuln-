static int bdrv_get_cluster_size(BlockDriverState *bs)

{

    BlockDriverInfo bdi;

    int ret;



    ret = bdrv_get_info(bs, &bdi);

    if (ret < 0 || bdi.cluster_size == 0) {

        return bs->request_alignment;

    } else {

        return bdi.cluster_size;

    }

}
