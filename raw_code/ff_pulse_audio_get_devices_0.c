int ff_pulse_audio_get_devices(AVDeviceInfoList *devices, const char *server, int output)

{

    pa_mainloop *pa_ml = NULL;

    pa_mainloop_api *pa_mlapi = NULL;

    pa_operation *pa_op = NULL;

    pa_context *pa_ctx = NULL;

    enum pa_operation_state op_state;

    enum PulseAudioContextState loop_state = PULSE_CONTEXT_INITIALIZING;

    PulseAudioDeviceList dev_list = { 0 };

    int i;



    dev_list.output = output;

    dev_list.devices = devices;

    if (!devices)

        return AVERROR(EINVAL);

    devices->nb_devices = 0;

    devices->devices = NULL;

    if (!(pa_ml = pa_mainloop_new()))

        return AVERROR(ENOMEM);

    if (!(pa_mlapi = pa_mainloop_get_api(pa_ml))) {

        dev_list.error_code = AVERROR_EXTERNAL;

        goto fail;

    }

    if (!(pa_ctx = pa_context_new(pa_mlapi, "Query devices"))) {

        dev_list.error_code = AVERROR(ENOMEM);

        goto fail;

    }

    pa_context_set_state_callback(pa_ctx, pa_state_cb, &loop_state);

    if (pa_context_connect(pa_ctx, server, 0, NULL) < 0) {

        dev_list.error_code = AVERROR_EXTERNAL;

        goto fail;

    }



    while (loop_state == PULSE_CONTEXT_INITIALIZING)

        pa_mainloop_iterate(pa_ml, 1, NULL);

    if (loop_state == PULSE_CONTEXT_FINISHED) {

        dev_list.error_code = AVERROR_EXTERNAL;

        goto fail;

    }



    if (output)

        pa_op = pa_context_get_sink_info_list(pa_ctx, pulse_audio_sink_device_cb, &dev_list);

    else

        pa_op = pa_context_get_source_info_list(pa_ctx, pulse_audio_source_device_cb, &dev_list);

    while ((op_state = pa_operation_get_state(pa_op)) == PA_OPERATION_RUNNING)

        pa_mainloop_iterate(pa_ml, 1, NULL);

    if (op_state != PA_OPERATION_DONE)

        dev_list.error_code = AVERROR_EXTERNAL;

    pa_operation_unref(pa_op);

    if (dev_list.error_code < 0)

        goto fail;



    pa_op = pa_context_get_server_info(pa_ctx, pulse_server_info_cb, &dev_list);

    while ((op_state = pa_operation_get_state(pa_op)) == PA_OPERATION_RUNNING)

        pa_mainloop_iterate(pa_ml, 1, NULL);

    if (op_state != PA_OPERATION_DONE)

        dev_list.error_code = AVERROR_EXTERNAL;

    pa_operation_unref(pa_op);

    if (dev_list.error_code < 0)

        goto fail;



    devices->default_device = -1;

    for (i = 0; i < devices->nb_devices; i++) {

        if (!strcmp(devices->devices[i]->device_name, dev_list.default_device)) {

            devices->default_device = i;

            break;

        }

    }



  fail:

    av_free(dev_list.default_device);

    if(pa_ctx)

        pa_context_disconnect(pa_ctx);

    if (pa_ctx)

        pa_context_unref(pa_ctx);

    if (pa_ml)

        pa_mainloop_free(pa_ml);

    return dev_list.error_code;

}
