static int map_f(BlockBackend *blk, int argc, char **argv)

{

    int64_t offset;

    int64_t nb_sectors, total_sectors;

    char s1[64];

    int64_t num;

    int ret;

    const char *retstr;



    offset = 0;

    total_sectors = blk_nb_sectors(blk);

    if (total_sectors < 0) {

        error_report("Failed to query image length: %s",

                     strerror(-total_sectors));

        return 0;

    }



    nb_sectors = total_sectors;



    do {

        ret = map_is_allocated(blk_bs(blk), offset, nb_sectors, &num);

        if (ret < 0) {

            error_report("Failed to get allocation status: %s", strerror(-ret));

            return 0;

        } else if (!num) {

            error_report("Unexpected end of image");

            return 0;

        }



        retstr = ret ? "    allocated" : "not allocated";

        cvtstr(offset << 9ULL, s1, sizeof(s1));

        printf("[% 24" PRId64 "] % 8" PRId64 "/% 8" PRId64 " sectors %s "

               "at offset %s (%d)\n",

               offset << 9ULL, num, nb_sectors, retstr, s1, ret);



        offset += num;

        nb_sectors -= num;

    } while (offset < total_sectors);



    return 0;

}
