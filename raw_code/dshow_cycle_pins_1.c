dshow_cycle_pins(AVFormatContext *avctx, enum dshowDeviceType devtype,

                 IBaseFilter *device_filter, IPin **ppin)

{

    struct dshow_ctx *ctx = avctx->priv_data;

    IEnumPins *pins = 0;

    IPin *device_pin = NULL;

    IPin *pin;

    int r;



    const GUID *mediatype[2] = { &MEDIATYPE_Video, &MEDIATYPE_Audio };

    const char *devtypename = (devtype == VideoDevice) ? "video" : "audio";



    int set_format = (devtype == VideoDevice && (ctx->video_size || ctx->framerate))

                  || (devtype == AudioDevice && (ctx->channels || ctx->sample_rate));

    int format_set = 0;



    r = IBaseFilter_EnumPins(device_filter, &pins);

    if (r != S_OK) {

        av_log(avctx, AV_LOG_ERROR, "Could not enumerate pins.\n");

        return AVERROR(EIO);

    }



    if (!ppin) {

        av_log(avctx, AV_LOG_INFO, "DirectShow %s device options\n",

               devtypename);

    }

    while (IEnumPins_Next(pins, 1, &pin, NULL) == S_OK && !device_pin) {

        IKsPropertySet *p = NULL;

        IEnumMediaTypes *types = NULL;

        PIN_INFO info = {0};

        AM_MEDIA_TYPE *type;

        GUID category;

        DWORD r2;



        IPin_QueryPinInfo(pin, &info);

        IBaseFilter_Release(info.pFilter);



        if (info.dir != PINDIR_OUTPUT)

            goto next;

        if (IPin_QueryInterface(pin, &IID_IKsPropertySet, (void **) &p) != S_OK)

            goto next;

        if (IKsPropertySet_Get(p, &AMPROPSETID_Pin, AMPROPERTY_PIN_CATEGORY,

                               NULL, 0, &category, sizeof(GUID), &r2) != S_OK)

            goto next;

        if (!IsEqualGUID(&category, &PIN_CATEGORY_CAPTURE))

            goto next;



        if (!ppin) {

            char *buf = dup_wchar_to_utf8(info.achName);

            av_log(avctx, AV_LOG_INFO, " Pin \"%s\"\n", buf);

            av_free(buf);

            dshow_cycle_formats(avctx, devtype, pin, NULL);

            goto next;

        }

        if (set_format) {

            dshow_cycle_formats(avctx, devtype, pin, &format_set);

            if (!format_set) {

                goto next;

            }

        }



        if (IPin_EnumMediaTypes(pin, &types) != S_OK)

            goto next;



        IEnumMediaTypes_Reset(types);

        while (IEnumMediaTypes_Next(types, 1, &type, NULL) == S_OK && !device_pin) {

            if (IsEqualGUID(&type->majortype, mediatype[devtype])) {

                device_pin = pin;

                goto next;

            }

            CoTaskMemFree(type);

        }



next:

        if (types)

            IEnumMediaTypes_Release(types);

        if (p)

            IKsPropertySet_Release(p);

        if (device_pin != pin)

            IPin_Release(pin);

    }



    IEnumPins_Release(pins);



    if (ppin) {

        if (set_format && !format_set) {

            av_log(avctx, AV_LOG_ERROR, "Could not set %s options\n", devtypename);

            return AVERROR(EIO);

        }

        if (!device_pin) {

            av_log(avctx, AV_LOG_ERROR,

                "Could not find output pin from %s capture device.\n", devtypename);

            return AVERROR(EIO);

        }

        *ppin = device_pin;

    }



    return 0;

}
