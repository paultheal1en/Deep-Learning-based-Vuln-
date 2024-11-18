static void test_smbios_ep_address(test_data *data)

{

    uint32_t off;



    /* find smbios entry point structure */

    for (off = 0xf0000; off < 0x100000; off += 0x10) {

        uint8_t sig[] = "_SM_";

        int i;



        for (i = 0; i < sizeof sig - 1; ++i) {

            sig[i] = readb(off + i);

        }



        if (!memcmp(sig, "_SM_", sizeof sig)) {

            break;

        }

    }



    g_assert_cmphex(off, <, 0x100000);

    data->smbios_ep_addr = off;

}
