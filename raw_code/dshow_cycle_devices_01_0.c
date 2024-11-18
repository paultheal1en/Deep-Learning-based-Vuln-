dshow_cycle_devices(AVFormatContext *avctx, ICreateDevEnum *devenum,

                    enum dshowDeviceType devtype, enum dshowSourceFilterType sourcetype, IBaseFilter **pfilter)

{

    struct dshow_ctx *ctx = avctx->priv_data;

    IBaseFilter *device_filter = NULL;

    IEnumMoniker *classenum = NULL;

    IMoniker *m = NULL;

    const char *device_name = ctx->device_name[devtype];

    int skip = (devtype == VideoDevice) ? ctx->video_device_number

                                        : ctx->audio_device_number;

    int r;



    const GUID *device_guid[2] = { &CLSID_VideoInputDeviceCategory,

                                   &CLSID_AudioInputDeviceCategory };

    const char *devtypename = (devtype == VideoDevice) ? "video" : "audio only";

    const char *sourcetypename = (sourcetype == VideoSourceDevice) ? "video" : "audio";



    r = ICreateDevEnum_CreateClassEnumerator(devenum, device_guid[sourcetype],

                                             (IEnumMoniker **) &classenum, 0);

    if (r != S_OK) {

        av_log(avctx, AV_LOG_ERROR, "Could not enumerate %s devices (or none found).\n",

               devtypename);

        return AVERROR(EIO);

    }



    while (!device_filter && IEnumMoniker_Next(classenum, 1, &m, NULL) == S_OK) {

        IPropertyBag *bag = NULL;

        char *friendly_name = NULL;

        char *unique_name = NULL;

        VARIANT var;

        IBindCtx *bind_ctx = NULL;

        LPOLESTR olestr = NULL;

        LPMALLOC co_malloc = NULL;

        int i;



        r = CoGetMalloc(1, &co_malloc);

        if (r = S_OK)

            goto fail1;

        r = CreateBindCtx(0, &bind_ctx);

        if (r != S_OK)

            goto fail1;

        /* GetDisplayname works for both video and audio, DevicePath doesn't */

        r = IMoniker_GetDisplayName(m, bind_ctx, NULL, &olestr);

        if (r != S_OK)

            goto fail1;

        unique_name = dup_wchar_to_utf8(olestr);

        /* replace ':' with '_' since we use : to delineate between sources */

        for (i = 0; i < strlen(unique_name); i++) {

            if (unique_name[i] == ':')

                unique_name[i] = '_';

        }



        r = IMoniker_BindToStorage(m, 0, 0, &IID_IPropertyBag, (void *) &bag);

        if (r != S_OK)

            goto fail1;



        var.vt = VT_BSTR;

        r = IPropertyBag_Read(bag, L"FriendlyName", &var, NULL);

        if (r != S_OK)

            goto fail1;

        friendly_name = dup_wchar_to_utf8(var.bstrVal);



        if (pfilter) {

            if (strcmp(device_name, friendly_name) && strcmp(device_name, unique_name))

                goto fail1;



            if (!skip--) {

                r = IMoniker_BindToObject(m, 0, 0, &IID_IBaseFilter, (void *) &device_filter);

                if (r != S_OK) {

                    av_log(avctx, AV_LOG_ERROR, "Unable to BindToObject for %s\n", device_name);

                    goto fail1;

                }

            }

        } else {

            av_log(avctx, AV_LOG_INFO, " \"%s\"\n", friendly_name);

            av_log(avctx, AV_LOG_INFO, "    Alternative name \"%s\"\n", unique_name);

        }



fail1:

        if (olestr && co_malloc)

            IMalloc_Free(co_malloc, olestr);

        if (bind_ctx)

            IBindCtx_Release(bind_ctx);

        av_free(friendly_name);

        av_free(unique_name);

        if (bag)

            IPropertyBag_Release(bag);

        IMoniker_Release(m);

    }



    IEnumMoniker_Release(classenum);



    if (pfilter) {

        if (!device_filter) {

            av_log(avctx, AV_LOG_ERROR, "Could not find %s device with name [%s] among source devices of type %s.\n",

                   devtypename, device_name, sourcetypename);

            return AVERROR(EIO);

        }

        *pfilter = device_filter;

    }



    return 0;

}
