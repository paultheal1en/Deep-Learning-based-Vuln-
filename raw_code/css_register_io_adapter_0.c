int css_register_io_adapter(CssIoAdapterType type, uint8_t isc, bool swap,

                            bool maskable, uint32_t *id)

{

    IoAdapter *adapter;

    bool found = false;

    int ret;

    S390FLICState *fs = s390_get_flic();

    S390FLICStateClass *fsc = S390_FLIC_COMMON_GET_CLASS(fs);



    *id = 0;

    QTAILQ_FOREACH(adapter, &channel_subsys.io_adapters, sibling) {

        if ((adapter->type == type) && (adapter->isc == isc)) {

            *id = adapter->id;

            found = true;

            ret = 0;

            break;

        }

        if (adapter->id >= *id) {

            *id = adapter->id + 1;

        }

    }

    if (found) {

        goto out;

    }

    adapter = g_new0(IoAdapter, 1);

    ret = fsc->register_io_adapter(fs, *id, isc, swap, maskable);

    if (ret == 0) {

        adapter->id = *id;

        adapter->isc = isc;

        adapter->type = type;

        QTAILQ_INSERT_TAIL(&channel_subsys.io_adapters, adapter, sibling);

    } else {

        g_free(adapter);

        fprintf(stderr, "Unexpected error %d when registering adapter %d\n",

                ret, *id);

    }

out:

    return ret;

}
