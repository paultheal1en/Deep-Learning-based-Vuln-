av_cold int vaapi_device_init(const char *device)
{
    int err;
    err = av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_VAAPI,
                                 device, NULL, 0);
    if (err < 0) {
        av_log(&vaapi_log, AV_LOG_ERROR, "Failed to create a VAAPI device\n");
        return err;
    }
    return 0;
}