static void virtio_setup(uint64_t dev_info)

{

    struct schib schib;

    int ssid;

    bool found = false;

    uint16_t dev_no;



    /*

     * We unconditionally enable mss support. In every sane configuration,

     * this will succeed; and even if it doesn't, stsch_err() can deal

     * with the consequences.

     */

    enable_mss_facility();



    if (dev_info != -1) {

        dev_no = dev_info & 0xffff;

        debug_print_int("device no. ", dev_no);

        blk_schid.ssid = (dev_info >> 16) & 0x3;

        debug_print_int("ssid ", blk_schid.ssid);

        found = find_dev(&schib, dev_no);

    } else {

        for (ssid = 0; ssid < 0x3; ssid++) {

            blk_schid.ssid = ssid;

            found = find_dev(&schib, -1);

            if (found) {

                break;

            }

        }

    }



    if (!found) {

        virtio_panic("No virtio-blk device found!\n");

    }



    virtio_setup_block(blk_schid);



    if (!virtio_ipl_disk_is_valid()) {

        virtio_panic("No valid hard disk detected.\n");

    }

}
