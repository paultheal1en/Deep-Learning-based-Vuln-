static int close_f(int argc, char **argv)

{

    bdrv_close(bs);

    bs = NULL;

    return 0;

}
