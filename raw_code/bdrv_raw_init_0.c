static void bdrv_raw_init(void)

{

    bdrv_register(&bdrv_raw);

    bdrv_register(&bdrv_host_device);

}
