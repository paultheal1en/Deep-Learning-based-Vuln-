static int nbd_co_writev(BlockDriverState *bs, int64_t sector_num,

                         int nb_sectors, QEMUIOVector *qiov)

{

    BDRVNBDState *s = bs->opaque;



    return nbd_client_session_co_writev(&s->client, sector_num,

                                        nb_sectors, qiov);

}
