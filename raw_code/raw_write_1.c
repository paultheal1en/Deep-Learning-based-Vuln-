static int raw_write(BlockDriverState *bs, int64_t sector_num,

                     const uint8_t *buf, int nb_sectors)

{

    if (check_write_unsafe(bs, sector_num, buf, nb_sectors)) {

        int ret;



        ret = raw_write_scrubbed_bootsect(bs, buf);

        if (ret < 0) {

            return ret;

        }



        ret = bdrv_write(bs->file, 1, buf + 512, nb_sectors - 1);

        if (ret < 0) {

            return ret;

        }



        return ret + 512;

    }



    return bdrv_write(bs->file, sector_num, buf, nb_sectors);

}
