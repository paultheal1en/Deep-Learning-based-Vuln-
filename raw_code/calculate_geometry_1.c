static int calculate_geometry(int64_t total_sectors, uint16_t* cyls,

    uint8_t* heads, uint8_t* secs_per_cyl)

{

    uint32_t cyls_times_heads;



    if (total_sectors > 65535 * 16 * 255)

        return -EFBIG;



    if (total_sectors > 65535 * 16 * 63) {

        *secs_per_cyl = 255;

        *heads = 16;

        cyls_times_heads = total_sectors / *secs_per_cyl;

    } else {

        *secs_per_cyl = 17;

        cyls_times_heads = total_sectors / *secs_per_cyl;

        *heads = (cyls_times_heads + 1023) / 1024;



        if (*heads < 4)

            *heads = 4;



        if (cyls_times_heads >= (*heads * 1024) || *heads > 16) {

            *secs_per_cyl = 31;

            *heads = 16;

            cyls_times_heads = total_sectors / *secs_per_cyl;

        }



        if (cyls_times_heads >= (*heads * 1024)) {

            *secs_per_cyl = 63;

            *heads = 16;

            cyls_times_heads = total_sectors / *secs_per_cyl;

        }

    }



    // Note: Rounding up deviates from the Virtual PC behaviour

    // However, we need this to avoid truncating images in qemu-img convert

    *cyls = (cyls_times_heads + *heads - 1) / *heads;



    return 0;

}
