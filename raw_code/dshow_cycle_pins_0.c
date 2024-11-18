dshow_cycle_pins(AVFormatContext *avctx, enum dshowDeviceType devtype,

                 IBaseFilter *device_filter, IPin **ppin)

{

    IEnumPins *pins = 0;

    IPin *device_pin = NULL;

    IPin *pin;

    int r;



    const GUID *mediatype[2] = { &MEDIATYPE_Video, &MEDIATYPE_Audio };

    const char *devtypename = (devtype == VideoDevice) ? "video" : "audio";



    r = IBaseFilter_EnumPins(device_filter, &pins);

    if (r != S_OK) {

        av_log(avctx, AV_LOG_ERROR, "Could not enumerate pins.\n");

        return AVERROR(EIO);

    }



    while (IEnumPins_Next(pins, 1, &pin, NULL) == S_OK && !device_pin) {

        IKsPropertySet *p = NULL;

        IEnumMediaTypes *types;

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



    if (!device_pin) {

        av_log(avctx, AV_LOG_ERROR,

               "Could not find output pin from %s capture device.\n", devtypename);

        return AVERROR(EIO);

    }

    *ppin = device_pin;



    return 0;

}
