static int vmdk_read(BlockDriverState *bs, int64_t sector_num,

                    uint8_t *buf, int nb_sectors)

{

    BDRVVmdkState *s = bs->opaque;

    int ret;

    uint64_t n, index_in_cluster;

    VmdkExtent *extent = NULL;

    uint64_t cluster_offset;



    while (nb_sectors > 0) {

        extent = find_extent(s, sector_num, extent);

        if (!extent) {

            return -EIO;

        }

        ret = get_cluster_offset(

                            bs, extent, NULL,

                            sector_num << 9, 0, &cluster_offset);

        index_in_cluster = sector_num % extent->cluster_sectors;

        n = extent->cluster_sectors - index_in_cluster;

        if (n > nb_sectors) {

            n = nb_sectors;

        }

        if (ret) {

            /* if not allocated, try to read from parent image, if exist */

            if (bs->backing_hd) {

                if (!vmdk_is_cid_valid(bs)) {

                    return -EINVAL;

                }

                ret = bdrv_read(bs->backing_hd, sector_num, buf, n);

                if (ret < 0) {

                    return ret;

                }

            } else {

                memset(buf, 0, 512 * n);

            }

        } else {

            ret = vmdk_read_extent(extent,

                            cluster_offset, index_in_cluster * 512,

                            buf, n);

            if (ret) {

                return ret;

            }

        }

        nb_sectors -= n;

        sector_num += n;

        buf += n * 512;

    }

    return 0;

}
