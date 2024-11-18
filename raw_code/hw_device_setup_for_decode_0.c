int hw_device_setup_for_decode(InputStream *ist)

{

    enum AVHWDeviceType type;

    HWDevice *dev;

    int err;



    if (ist->hwaccel_device) {

        dev = hw_device_get_by_name(ist->hwaccel_device);

        if (!dev) {

            char *tmp;

            type = hw_device_match_type_by_hwaccel(ist->hwaccel_id);

            if (type == AV_HWDEVICE_TYPE_NONE) {

                // No match - this isn't necessarily invalid, though,

                // because an explicit device might not be needed or

                // the hwaccel setup could be handled elsewhere.

                return 0;

            }

            tmp = av_asprintf("%s:%s", av_hwdevice_get_type_name(type),

                              ist->hwaccel_device);

            if (!tmp)

                return AVERROR(ENOMEM);

            err = hw_device_init_from_string(tmp, &dev);

            av_free(tmp);

            if (err < 0)

                return err;

        }

    } else {

        if (ist->hwaccel_id != HWACCEL_NONE)

            type = hw_device_match_type_by_hwaccel(ist->hwaccel_id);

        else

            type = hw_device_match_type_in_name(ist->dec->name);

        if (type != AV_HWDEVICE_TYPE_NONE) {

            dev = hw_device_get_by_type(type);

            if (!dev) {

                hw_device_init_from_string(av_hwdevice_get_type_name(type),

                                           &dev);

            }

        } else {

            // No device required.

            return 0;

        }

    }



    if (!dev) {

        av_log(ist->dec_ctx, AV_LOG_WARNING, "No device available "

               "for decoder (device type %s for codec %s).\n",

               av_hwdevice_get_type_name(type), ist->dec->name);

        return 0;

    }



    ist->dec_ctx->hw_device_ctx = av_buffer_ref(dev->device_ref);

    if (!ist->dec_ctx->hw_device_ctx)

        return AVERROR(ENOMEM);



    return 0;

}
