static int qcow_read(BlockDriverState *bs, int64_t sector_num,

                     uint8_t *buf, int nb_sectors)

{

    BDRVQcowState *s = bs->opaque;

    int ret, index_in_cluster, n, n1;

    uint64_t cluster_offset;



    while (nb_sectors > 0) {

        n = nb_sectors;

        cluster_offset = qcow2_get_cluster_offset(bs, sector_num << 9, &n);

        index_in_cluster = sector_num & (s->cluster_sectors - 1);

        if (!cluster_offset) {

            if (bs->backing_hd) {

                /* read from the base image */

                n1 = qcow2_backing_read1(bs->backing_hd, sector_num, buf, n);

                if (n1 > 0) {

                    BLKDBG_EVENT(bs->file, BLKDBG_READ_BACKING);

                    ret = bdrv_read(bs->backing_hd, sector_num, buf, n1);

                    if (ret < 0)

                        return -1;

                }

            } else {

                memset(buf, 0, 512 * n);

            }

        } else if (cluster_offset & QCOW_OFLAG_COMPRESSED) {

            if (qcow2_decompress_cluster(bs, cluster_offset) < 0)

                return -1;

            memcpy(buf, s->cluster_cache + index_in_cluster * 512, 512 * n);

        } else {

            BLKDBG_EVENT(bs->file, BLKDBG_READ);

            ret = bdrv_pread(bs->file, cluster_offset + index_in_cluster * 512, buf, n * 512);

            if (ret != n * 512)

                return -1;

            if (s->crypt_method) {

                qcow2_encrypt_sectors(s, sector_num, buf, buf, n, 0,

                                &s->aes_decrypt_key);

            }

        }

        nb_sectors -= n;

        sector_num += n;

        buf += n * 512;

    }

    return 0;

}
