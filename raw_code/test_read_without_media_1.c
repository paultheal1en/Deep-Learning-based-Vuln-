static void test_read_without_media(void)

{

    uint8_t ret;



    ret = send_read_command();

    g_assert(ret == 0);

}
