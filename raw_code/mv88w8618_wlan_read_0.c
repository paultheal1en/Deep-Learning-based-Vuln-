static uint64_t mv88w8618_wlan_read(void *opaque, target_phys_addr_t offset,

                                    unsigned size)

{

    switch (offset) {

    /* Workaround to allow loading the binary-only wlandrv.ko crap

     * from the original Freecom firmware. */

    case MP_WLAN_MAGIC1:

        return ~3;

    case MP_WLAN_MAGIC2:

        return -1;



    default:

        return 0;

    }

}
