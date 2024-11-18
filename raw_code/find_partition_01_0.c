static int find_partition(BlockBackend *blk, int partition,

                          off_t *offset, off_t *size)

{

    struct partition_record mbr[4];

    uint8_t data[512];

    int i;

    int ext_partnum = 4;

    int ret;



    if ((ret = blk_read(blk, 0, data, 1)) < 0) {

        error_report("error while reading: %s", strerror(-ret));

        exit(EXIT_FAILURE);

    }



    if (data[510] != 0x55 || data[511] != 0xaa) {

        return -EINVAL;

    }



    for (i = 0; i < 4; i++) {

        read_partition(&data[446 + 16 * i], &mbr[i]);



        if (!mbr[i].system || !mbr[i].nb_sectors_abs) {

            continue;

        }



        if (mbr[i].system == 0xF || mbr[i].system == 0x5) {

            struct partition_record ext[4];

            uint8_t data1[512];

            int j;



            if ((ret = blk_read(blk, mbr[i].start_sector_abs, data1, 1)) < 0) {

                error_report("error while reading: %s", strerror(-ret));

                exit(EXIT_FAILURE);

            }



            for (j = 0; j < 4; j++) {

                read_partition(&data1[446 + 16 * j], &ext[j]);

                if (!ext[j].system || !ext[j].nb_sectors_abs) {

                    continue;

                }



                if ((ext_partnum + j + 1) == partition) {

                    *offset = (uint64_t)ext[j].start_sector_abs << 9;

                    *size = (uint64_t)ext[j].nb_sectors_abs << 9;

                    return 0;

                }

            }

            ext_partnum += 4;

        } else if ((i + 1) == partition) {

            *offset = (uint64_t)mbr[i].start_sector_abs << 9;

            *size = (uint64_t)mbr[i].nb_sectors_abs << 9;

            return 0;

        }

    }



    return -ENOENT;

}
