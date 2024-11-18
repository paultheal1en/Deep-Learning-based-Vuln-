static int print_device_sources(AVInputFormat *fmt, AVDictionary *opts)

{

    int ret, i;

    AVFormatContext *dev = NULL;

    AVDeviceInfoList *device_list = NULL;

    AVDictionary *tmp_opts = NULL;



    if (!fmt || !fmt->priv_class  || !AV_IS_INPUT_DEVICE(fmt->priv_class->category))

        return AVERROR(EINVAL);



    printf("Audo-detected sources for %s:\n", fmt->name);

    if (!fmt->get_device_list) {

        ret = AVERROR(ENOSYS);

        printf("Cannot list sources. Not implemented.\n");

        goto fail;

    }



    /* TODO: avformat_open_input calls read_header callback which is not necessary.

             Function like avformat_alloc_output_context2 for input could be helpful here. */

    av_dict_copy(&tmp_opts, opts, 0);

    if ((ret = avformat_open_input(&dev, NULL, fmt, &tmp_opts)) < 0) {

        printf("Cannot open device: %s.\n", fmt->name);

        goto fail;

    }



    if ((ret = avdevice_list_devices(dev, &device_list)) < 0) {

        printf("Cannot list sources.\n");

        goto fail;

    }



    for (i = 0; i < device_list->nb_devices; i++) {

        printf("%s %s [%s]\n", device_list->default_device == i ? "*" : " ",

               device_list->devices[i]->device_name, device_list->devices[i]->device_description);

    }



  fail:

    av_dict_free(&tmp_opts);

    avdevice_free_list_devices(&device_list);

    avformat_close_input(&dev);

    return ret;

}
