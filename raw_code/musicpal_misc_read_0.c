static uint64_t musicpal_misc_read(void *opaque, target_phys_addr_t offset,

                                   unsigned size)

{

    switch (offset) {

    case MP_MISC_BOARD_REVISION:

        return MP_BOARD_REVISION;



    default:

        return 0;

    }

}
