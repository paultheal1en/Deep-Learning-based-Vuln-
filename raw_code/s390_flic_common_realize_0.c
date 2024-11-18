static void s390_flic_common_realize(DeviceState *dev, Error **errp)

{

    S390FLICState *fs = S390_FLIC_COMMON(dev);

    uint32_t max_batch = fs->adapter_routes_max_batch;



    if (max_batch > ADAPTER_ROUTES_MAX_GSI) {

        error_setg(errp, "flic property adapter_routes_max_batch too big"

                   " (%d > %d)", max_batch, ADAPTER_ROUTES_MAX_GSI);

    }



    fs->ais_supported = true;

}
