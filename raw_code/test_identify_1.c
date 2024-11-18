static void test_identify(void)

{

    uint8_t data;

    uint16_t buf[256];

    int i;

    int ret;



    ide_test_start(

        "-vnc none "

        "-drive file=%s,if=ide,serial=%s,cache=writeback "

        "-global ide-hd.ver=%s",

        tmp_path, "testdisk", "version");



    /* IDENTIFY command on device 0*/

    outb(IDE_BASE + reg_device, 0);

    outb(IDE_BASE + reg_command, CMD_IDENTIFY);



    /* Read in the IDENTIFY buffer and check registers */

    data = inb(IDE_BASE + reg_device);

    g_assert_cmpint(data & DEV, ==, 0);



    for (i = 0; i < 256; i++) {

        data = inb(IDE_BASE + reg_status);

        assert_bit_set(data, DRDY | DRQ);

        assert_bit_clear(data, BSY | DF | ERR);



        ((uint16_t*) buf)[i] = inw(IDE_BASE + reg_data);

    }



    data = inb(IDE_BASE + reg_status);

    assert_bit_set(data, DRDY);

    assert_bit_clear(data, BSY | DF | ERR | DRQ);



    /* Check serial number/version in the buffer */

    string_cpu_to_be16(&buf[10], 20);

    ret = memcmp(&buf[10], "testdisk            ", 20);

    g_assert(ret == 0);



    string_cpu_to_be16(&buf[23], 8);

    ret = memcmp(&buf[23], "version ", 8);

    g_assert(ret == 0);



    /* Write cache enabled bit */

    assert_bit_set(buf[85], 0x20);



    ide_test_quit();

}
