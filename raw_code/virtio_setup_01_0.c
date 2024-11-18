static void virtio_setup(uint64_t dev_info)

{

    struct schib schib;

    int i;

    int r;

    bool found = false;

    bool check_devno = false;

    uint16_t dev_no = -1;

    blk_schid.one = 1;



    if (dev_info != -1) {

        check_devno = true;

        dev_no = dev_info & 0xffff;

        debug_print_int("device no. ", dev_no);

        blk_schid.ssid = (dev_info >> 16) & 0x3;

        if (blk_schid.ssid != 0) {

            debug_print_int("ssid ", blk_schid.ssid);

            if (enable_mss_facility() != 0) {

                virtio_panic("Failed to enable mss facility\n");

            }

        }

    }



    for (i = 0; i < 0x10000; i++) {

        blk_schid.sch_no = i;

        r = stsch_err(blk_schid, &schib);

        if (r == 3) {

            break;

        }

        if (schib.pmcw.dnv) {

            if (!check_devno || (schib.pmcw.dev == dev_no)) {

                if (virtio_is_blk(blk_schid)) {

                    found = true;

                    break;

                }

            }

        }

    }



    if (!found) {

        virtio_panic("No virtio-blk device found!\n");

    }



    virtio_setup_block(blk_schid);

}
