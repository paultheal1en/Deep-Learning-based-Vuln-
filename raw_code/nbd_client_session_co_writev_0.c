int nbd_client_session_co_writev(NbdClientSession *client, int64_t sector_num,

                                 int nb_sectors, QEMUIOVector *qiov)

{

    int offset = 0;

    int ret;

    while (nb_sectors > NBD_MAX_SECTORS) {

        ret = nbd_co_writev_1(client, sector_num,

                              NBD_MAX_SECTORS, qiov, offset);

        if (ret < 0) {

            return ret;

        }

        offset += NBD_MAX_SECTORS * 512;

        sector_num += NBD_MAX_SECTORS;

        nb_sectors -= NBD_MAX_SECTORS;

    }

    return nbd_co_writev_1(client, sector_num, nb_sectors, qiov, offset);

}
