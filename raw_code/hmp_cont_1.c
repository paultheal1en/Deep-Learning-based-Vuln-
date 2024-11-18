void hmp_cont(Monitor *mon, const QDict *qdict)

{

    BlockInfoList *bdev_list, *bdev;

    Error *err = NULL;



    bdev_list = qmp_query_block(NULL);

    for (bdev = bdev_list; bdev; bdev = bdev->next) {

        if (key_is_missing(bdev->value)) {

            monitor_read_block_device_key(mon, bdev->value->device,

                                          hmp_cont_cb, NULL);

            goto out;

        }

    }



    qmp_cont(&err);

    hmp_handle_error(mon, &err);



out:

    qapi_free_BlockInfoList(bdev_list);

}
