static int enable_write_target(BDRVVVFATState *s)

{

    BlockDriver *bdrv_qcow;

    QEMUOptionParameter *options;

    Error *local_err = NULL;

    int ret;

    int size = sector2cluster(s, s->sector_count);

    s->used_clusters = calloc(size, 1);



    array_init(&(s->commits), sizeof(commit_t));



    s->qcow_filename = g_malloc(1024);

    ret = get_tmp_filename(s->qcow_filename, 1024);

    if (ret < 0) {

        goto err;

    }



    bdrv_qcow = bdrv_find_format("qcow");

    options = parse_option_parameters("", bdrv_qcow->create_options, NULL);

    set_option_parameter_int(options, BLOCK_OPT_SIZE, s->sector_count * 512);

    set_option_parameter(options, BLOCK_OPT_BACKING_FILE, "fat:");



    ret = bdrv_create(bdrv_qcow, s->qcow_filename, options, &local_err);

    if (ret < 0) {

        qerror_report_err(local_err);

        error_free(local_err);

        goto err;

    }



    s->qcow = NULL;

    ret = bdrv_open(&s->qcow, s->qcow_filename, NULL, NULL,

            BDRV_O_RDWR | BDRV_O_CACHE_WB | BDRV_O_NO_FLUSH, bdrv_qcow,

            &local_err);

    if (ret < 0) {

        qerror_report_err(local_err);

        error_free(local_err);

        goto err;

    }



#ifndef _WIN32

    unlink(s->qcow_filename);

#endif



    s->bs->backing_hd = bdrv_new("");

    s->bs->backing_hd->drv = &vvfat_write_target;

    s->bs->backing_hd->opaque = g_malloc(sizeof(void*));

    *(void**)s->bs->backing_hd->opaque = s;



    return 0;



err:

    g_free(s->qcow_filename);

    s->qcow_filename = NULL;

    return ret;

}
