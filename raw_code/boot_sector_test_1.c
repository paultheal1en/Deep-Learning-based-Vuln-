void boot_sector_test(void)

{

    uint8_t signature_low;

    uint8_t signature_high;

    uint16_t signature;

    int i;



    /* Wait at most 90 seconds */

#define TEST_DELAY (1 * G_USEC_PER_SEC / 10)

#define TEST_CYCLES MAX((90 * G_USEC_PER_SEC / TEST_DELAY), 1)



    /* Poll until code has run and modified memory.  Once it has we know BIOS

     * initialization is done.  TODO: check that IP reached the halt

     * instruction.

     */

    for (i = 0; i < TEST_CYCLES; ++i) {

        signature_low = readb(BOOT_SECTOR_ADDRESS + SIGNATURE_OFFSET);

        signature_high = readb(BOOT_SECTOR_ADDRESS + SIGNATURE_OFFSET + 1);

        signature = (signature_high << 8) | signature_low;

        if (signature == SIGNATURE) {

            break;

        }

        g_usleep(TEST_DELAY);

    }



    g_assert_cmphex(signature, ==, SIGNATURE);

}
