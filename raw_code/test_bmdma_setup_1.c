static void test_bmdma_setup(void)

{

    ide_test_start(

        "-vnc none "

        "-drive file=%s,if=ide,serial=%s,cache=writeback "

        "-global ide-hd.ver=%s",

        tmp_path, "testdisk", "version");

}
