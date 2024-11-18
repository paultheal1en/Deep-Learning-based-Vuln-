void hmp_info_block(Monitor *mon, const QDict *qdict)

{

    BlockInfoList *block_list, *info;

    ImageInfo *image_info;

    const char *device = qdict_get_try_str(qdict, "device");

    bool verbose = qdict_get_try_bool(qdict, "verbose", 0);



    block_list = qmp_query_block(NULL);



    for (info = block_list; info; info = info->next) {

        if (device && strcmp(device, info->value->device)) {

            continue;

        }

        monitor_printf(mon, "%s: removable=%d",

                       info->value->device, info->value->removable);



        if (info->value->removable) {

            monitor_printf(mon, " locked=%d", info->value->locked);

            monitor_printf(mon, " tray-open=%d", info->value->tray_open);

        }



        if (info->value->has_io_status) {

            monitor_printf(mon, " io-status=%s",

                           BlockDeviceIoStatus_lookup[info->value->io_status]);

        }



        if (info->value->has_inserted) {

            monitor_printf(mon, " file=");

            monitor_print_filename(mon, info->value->inserted->file);



            if (info->value->inserted->has_backing_file) {

                monitor_printf(mon, " backing_file=");

                monitor_print_filename(mon, info->value->inserted->backing_file);

                monitor_printf(mon, " backing_file_depth=%" PRId64,

                    info->value->inserted->backing_file_depth);

            }

            monitor_printf(mon, " ro=%d drv=%s encrypted=%d",

                           info->value->inserted->ro,

                           info->value->inserted->drv,

                           info->value->inserted->encrypted);



            monitor_printf(mon, " bps=%" PRId64 " bps_rd=%" PRId64

                            " bps_wr=%" PRId64 " iops=%" PRId64

                            " iops_rd=%" PRId64 " iops_wr=%" PRId64,

                            info->value->inserted->bps,

                            info->value->inserted->bps_rd,

                            info->value->inserted->bps_wr,

                            info->value->inserted->iops,

                            info->value->inserted->iops_rd,

                            info->value->inserted->iops_wr);



            if (verbose) {

                monitor_printf(mon, " images:\n");

                image_info = info->value->inserted->image;

                while (1) {

                        bdrv_image_info_dump((fprintf_function)monitor_printf,

                                             mon, image_info);

                    if (image_info->has_backing_image) {

                        image_info = image_info->backing_image;

                    } else {

                        break;

                    }

                }

            }

        } else {

            monitor_printf(mon, " [not inserted]");

        }



        monitor_printf(mon, "\n");

    }



    qapi_free_BlockInfoList(block_list);

}
