int bdrv_create_file(const char* filename, QEMUOptionParameter *options)

{

    BlockDriver *drv;



    drv = bdrv_find_protocol(filename);

    if (drv == NULL) {

        drv = bdrv_find_format("file");

    }



    return bdrv_create(drv, filename, options);

}
