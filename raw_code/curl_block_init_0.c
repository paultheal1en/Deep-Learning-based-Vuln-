static void curl_block_init(void)

{

    bdrv_register(&bdrv_http);

    bdrv_register(&bdrv_https);

    bdrv_register(&bdrv_ftp);

    bdrv_register(&bdrv_ftps);

    bdrv_register(&bdrv_tftp);

}
